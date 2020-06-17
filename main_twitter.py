import torch
import time
from data import load_dataset
from models import StyleTransformer, Discriminator
from train import train, auto_eval
import pickle

class Config():
    data_path = './data_twitter/'
    log_dir = 'runs/exp'
    save_path = './save'
    pretrained_embed_path = './embedding/'
    device = torch.device('cuda' if True and torch.cuda.is_available() else 'cpu')
    discriminator_method = 'Multi' # 'Multi' or 'Cond'
    load_pretrained_embed = False
    min_freq = 3
    max_length = 40
    embed_size = 128
    d_model = 256
    h = 4
    num_styles = 2
    num_classes = num_styles + 1 if discriminator_method == 'Multi' else 2
    num_layers = 4
    batch_size = 10
    lr_F = 0.0001
    lr_D = 0.0001
    L2 = 0
    iter_D = 10
    iter_F = 5
    F_pretrain_iter = 500
    log_steps = 5
    eval_steps = 25
    learned_pos_embed = True
    dropout = 0
    drop_rate_config = [(1, 0)]
    temperature_config = [(1, 0)]

    slf_factor = 0.25
    cyc_factor = 0.5
    adv_factor = 1

    inp_shuffle_len = 0
    inp_unk_drop_fac = 0
    inp_rand_drop_fac = 0
    inp_drop_prob = 0
    
    acc_path = 'fasttext_twitter.bin'
    ppl_path = 'twitter_5_kenlm.binary'
    pos_label = '__label__pos'


def main():
    config = Config()
    train_iters, dev_iters, test_iters, vocab = load_dataset(config,
                                                            train_pos='twitter_pos.train', train_neg='twitter_neg.train',
                                                            dev_pos='twitter_pos.dev', dev_neg='twitter_neg.dev',
                                                            test_pos='twitter_pos.test', test_neg='twitter_neg.test')
    print('Vocab size:', len(vocab))
    model_F = StyleTransformer(config, vocab).to(config.device)
    model_D = Discriminator(config, vocab).to(config.device)
    print(config.discriminator_method)
    
    train(config, vocab, model_F, model_D, train_iters, dev_iters, test_iters)
    torch.save(model_F.state_dict(), 'modelF_trained')
    torch.save(model_D.state_dict(), 'modelD_trained')

if __name__ == '__main__':
    main()
