import os
import random
import argparse
import yaml
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.nn as nn

from datasets.imagenet import ImageNet
import clip
from utils import *
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()
def get_arguments():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config', help='settings of Tip-Adapter in yaml format')
    args = parser.parse_args()

    return args

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)       

class CoOp_PromptLearner(nn.Module):
    def __init__(self, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = 4
        ctx_init = 'a photo of a' # caltech101 
#         ctx_init = None
        self.dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        
        self.n_cls = n_cls
        self.n_ctx = n_ctx

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init).cuda()
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(self.dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :].cuda()
            prompt_prefix = ctx_init       
            self.n_ctx = n_ctx
        else:
            # random initialization
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=self.dtype).cuda()
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)
        self.prompt_prefix = prompt_prefix
        self.get_prefix_suffix_token(classnames, clip_model)

        
    def get_prefix_suffix_token(self, classnames, clip_model):
        prompt_prefix = self.prompt_prefix
        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).cuda()  # (n_cls, n_tkn)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(self.dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + self.n_ctx :, :])  # CLS, EOS
        
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        prompts = torch.cat(
            [
                prefix,  # (n_cls, 1, dim)
                ctx,     # (n_cls, n_ctx, dim)
                suffix,  # (n_cls, *, dim)
            ],
            dim=1,
        )

        return prompts
    

def run_coop(cfg, text_encoder, prompt_learner, clip_weights, clip_model, netG=None):  
    coop_prompt_learner = CoOp_PromptLearner(all_classnames, clip_model)
#     optimizer = torch.optim.SGD(coop_prompt_learner.parameters(), lr=2e-3)
    optimizer = torch.optim.AdamW(coop_prompt_learner.parameters(), lr=1e-3, eps=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg['train_epoch'] * len(train_loader_F))
    
    best_base_acc, best_new_acc, best_H = 0, 0, 0
    best_epoch = 0
    
    for train_idx in range(cfg['train_epoch']):
        # Train
        loss_list = []
        print('Train Epoch: {:} / {:}'.format(train_idx, cfg['train_epoch']))

        for i, (images, target) in enumerate(tqdm(train_loader_F)):
            images, target = images.cuda(), target.cuda()
            
            with torch.no_grad():
                image_features = clip_model.encode_image(images)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                
            if netG is not None:
                with torch.no_grad():
                    gen_target = torch.randint(len(base_classnames), len(all_classnames), (target.shape[0], )).cuda()
                    z = torch.randn([gen_target.shape[0], image_features.shape[1]]).cuda()              
                    text_features = clip_weights.T[gen_target].float()
                    bias = netG(z)
                    prompt_learner.get_prefix_suffix_token(all_classnames, clip_model) # update prefix and suffix for new dataset.
                    prompts = prompt_learner(bias, gen_target) 
                    tokenized_prompts = prompt_learner.tokenized_prompts
                    text_features = text_encoder(prompts, tokenized_prompts[gen_target])
                    gen_feature = text_features / text_features.norm(dim=-1, keepdim=True)
                    gen_target = gen_target 
                image_features = torch.cat([image_features, gen_feature], dim=0).half()
                target = torch.cat([target, gen_target], dim=0).half()
            
            prompts = coop_prompt_learner()
            tokenized_prompts = coop_prompt_learner.tokenized_prompts

            text_features = text_encoder(prompts, tokenized_prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            logits = 100. * image_features.float() @ text_features.T.float()
            loss = F.cross_entropy(logits, target.long())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
    # Evaluation
    with torch.no_grad():
        prompts = coop_prompt_learner()
        tokenized_prompts = coop_prompt_learner.tokenized_prompts
        text_features = text_encoder(prompts, tokenized_prompts)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # new
        clip_logits = 100. * test_features_new.float() @ text_features.T.float()[:, len(base_classnames):]
        new_acc = cls_acc(clip_logits, test_labels_new)

        # base
        clip_logits = 100. * test_features.float() @ text_features.T.float()[:, :len(base_classnames)]
        base_acc = cls_acc(clip_logits, test_labels)

        H = 2 * base_acc * new_acc / (base_acc + new_acc)
        if H > best_H:
            best_base_acc = base_acc
            best_new_acc = new_acc
            best_H = H
            best_epoch = train_idx

    print("base acc:\t%.2f  new acc:\t%.2f H:\t%.2f " % (base_acc, new_acc, H))
                
    print(f"**** After fine-tuning, CoOp's best base test accuracy: {best_base_acc:.2f}, at epoch: {best_epoch}. ****\n")
    print(f"**** After fine-tuning, CoOp's best new test accuracy: {best_new_acc:.2f}, at epoch: {best_epoch}. ****\n")
    print(f"**** After fine-tuning, CoOp's best H test accuracy: {best_H:.2f} ****\n")
    
    return best_base_acc, best_new_acc, best_H

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts.half() + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x.float()).half() # LayerNorm need to compute at fp32 for fp16 input
        x = x.permute(1, 0, 2)  # LND -> NLD     
        x = self.ln_final(x.float()).type(self.dtype)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection.type(self.dtype)
        return x
    
class PromptLearner(nn.Module):
    def __init__(self, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = 4  
#         ctx_init = 'a photo of a'
        ctx_init = None
        self.dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        
        self.n_cls = n_cls
        self.n_ctx = n_ctx

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init).cuda()
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(self.dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :].cuda()
            prompt_prefix = ctx_init       
            self.n_ctx = n_ctx
        else:
            # random initialization
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=self.dtype).cuda()
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)
        self.prompt_prefix = prompt_prefix
        self.get_prefix_suffix_token(classnames, clip_model)
        
    def get_prefix_suffix_token(self, classnames, clip_model):
        prompt_prefix = self.prompt_prefix
        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).cuda()  # (n_cls, n_tkn)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(self.dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + self.n_ctx :, :])  # CLS, EOS
        
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens

    def forward(self, bias, target):
        prefix = self.token_prefix[target]
        suffix = self.token_suffix[target]
        ctx = self.ctx                     # (n_ctx, ctx_dim)
        bias = bias.unsqueeze(1)           # (batch, 1, ctx_dim)
        ctx = ctx.unsqueeze(0)             # (1, n_ctx, ctx_dim)
        ctx_shifted = ctx + bias           # (batch, n_ctx, ctx_dim)
        prompts = torch.cat([prefix, ctx_shifted, suffix], dim=1)
        return prompts
    
def vae_loss(recon_x, x, mean, log_var, target, clip_weights):
    REC = (recon_x - x).pow(2).sum(1).mean()
    KLD = -0.5 * (1 + log_var - mean.pow(2) - log_var.exp()).sum(dim=1).mean()
    return (REC + 1 * KLD)

class Encoder(nn.Module):
    
    def __init__(self):
        super(Encoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(512 * 1, 4096),
            nn.ReLU(),
        )
        self.mean = nn.Linear(4096, 512)
        self.log_var = nn.Linear(4096, 512)
        self.apply(weights_init)
        
    def forward(self, x, a):
#         x = torch.cat([x, a], dim=1)
        x = self.net(x)
        mean = self.mean(x)
        log_var = self.log_var(x)
        return mean, log_var

class Generator(nn.Module):
    
    def __init__(self):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(512 * 1, 4096),
            nn.LeakyReLU(0.2),
            nn.Linear(4096, 512)
        )
        self.apply(weights_init)
    
    def forward(self, x):
        out = self.net(x)
        return out
    
def run_vae_generator(cfg, clip_weights, clip_model): 
    # CLIP
    for p in clip_model.parameters():
        p.requires_grad = False
    
    text_encoder = TextEncoder(clip_model).float().cuda()
    prompt_learner = PromptLearner(all_classnames, clip_model).float().cuda()
    
    # train VAE.
    netE = Encoder().cuda()
    netG = Generator().cuda()
    optimizerE = torch.optim.AdamW(netE.parameters(), lr=1e-3)
    optimizerG = torch.optim.AdamW(netG.parameters(), lr=1e-3)
    optimizerP = torch.optim.AdamW(prompt_learner.parameters(), lr=1e-3)
    
    best_base, best_new, best_H = 0.0, 0.0, 0.0
    
    for train_idx in range(1, 10 + 1):       
        # Train
        netE.train()
        netG.train()
        
        loss_list = []
        print('Train Epoch: {:} / {:}'.format(train_idx, cfg['train_epoch']))
        
        for i, (images, target) in enumerate(tqdm(train_loader_F)):
            images, target = images.cuda(), target.cuda()
            with torch.no_grad():
                image_features = clip_model.encode_image(images).float()
                image_features /= image_features.norm(dim=-1, keepdim=True)
            
            text_features = clip_weights.T[target].float()
            netE.zero_grad()
            netG.zero_grad()
            mean, log_var = netE(image_features, text_features)
            std = torch.exp(0.5 * log_var)
            z = torch.randn(mean.shape).cuda()
            z = std * z + mean
            bias = netG(z)
            prompt_learner.get_prefix_suffix_token(base_classnames, clip_model)
            prompts = prompt_learner(bias, target)
            tokenized_prompts = prompt_learner.tokenized_prompts
            text_features = text_encoder(prompts, tokenized_prompts[target])
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)      
            recon_features = text_features
            loss = vae_loss(recon_features, image_features, mean, log_var, target, clip_weights)
            optimizerP.zero_grad()
            loss.backward()
            loss_list.append(loss.item())
            optimizerE.step()
            optimizerG.step()
            optimizerP.step()

        print('Loss: {:.4f}'.format(sum(loss_list)/len(loss_list)))
        if train_idx % 10 == 0:
            # Evaluation.
            netE.eval()
            netG.eval()
            clip_weights_mix = torch.cat([clip_weights, clip_weights_new], dim=1)
            base, new, H = run_coop(cfg, text_encoder, prompt_learner, clip_weights_mix, clip_model, netG=netG)
            if H > best_H:
                best_base = base
                best_new = new
                best_H = H
                best_epoch = train_idx
            print("base acc:\t%.2f  new acc:\t%.2f H:\t%.2f " % (base, new, H))
         
    print("Evaluate on dataset:", cfg['dataset'])
    print("best base acc: %.2f" % best_base)
    print("best new acc: %.2f" % best_new)
    print("best H: %.2f" % best_H)

def main():

    # Load config file
    args = get_arguments()
    assert (os.path.exists(args.config))
    
    cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    cache_dir = os.path.join('./caches', cfg['dataset'])
    os.makedirs(cache_dir, exist_ok=True)
    cfg['cache_dir'] = cache_dir

    print("\nRunning configs.")
    print(cfg, "\n")

    # CLIP
    clip_model, preprocess = clip.load(cfg['backbone'])
    clip_model.eval()

    # ImageNet dataset
    random.seed(1)
    torch.manual_seed(1)
    
    global test_features, test_labels, train_loader_F
    global test_features_new, test_labels_new, train_loader_F_new
    global base_classnames, new_classnames, all_classnames
    global clip_weights_new
    
    print("Preparing ImageNet dataset.")
    # base classses
    cfg['subsample_classes'] = 'base' # all/base/new
    imagenet = ImageNet(cfg, cfg['root_path'], cfg['shots'], preprocess)
    test_loader = torch.utils.data.DataLoader(imagenet.test, batch_size=64, num_workers=8, shuffle=False)
    train_loader_F = torch.utils.data.DataLoader(imagenet.train, batch_size=256, num_workers=8, shuffle=True)
    # Textual features
    print("Getting textual features as CLIP's classifier.")
    clip_weights = clip_classifier(imagenet.classnames, imagenet.template, clip_model)
    base_classnames = imagenet.classnames

    # Pre-load test features
    print("\nLoading visual features and labels from test set.")
    test_features, test_labels = pre_load_features(cfg, "test", clip_model, test_loader)
    
    # new classes
    cfg['subsample_classes'] = 'new' # all/base/new
    imagenet = ImageNet(cfg, cfg['root_path'], cfg['shots'], preprocess)
    test_loader = torch.utils.data.DataLoader(imagenet.test, batch_size=64, num_workers=8, shuffle=False)
    test_features_new, test_labels_new = pre_load_features(cfg, "test", clip_model, test_loader)
    test_labels_new = test_labels_new - 500
    clip_weights_new = clip_classifier(imagenet.classnames, imagenet.template, clip_model)
    new_classnames = imagenet.classnames
    
    all_classnames = base_classnames + new_classnames

    # 
    run_vae_generator(cfg, clip_weights, clip_model)
           

if __name__ == '__main__':
    main()
