import numpy as np
import time
import torch
import random
from torch.utils.data import DataLoader
from torch import nn, Tensor
from tempfile import TemporaryDirectory
import os
import pickle

from lr_scheduler.warmup_reduce_lr_on_plateau_scheduler import WarmupReduceLROnPlateauScheduler
from model import Causalformer

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--datafolder', type=str, default='../izhikevich_data/')  # data source folder
    parser.add_argument('--expfolder', type=str, default='exc4_inh1_nodiag_withinp_rand/p2/inpvar5_S5_t5000/seed5/')  # experiment folder
    parser.add_argument('--datafile', type=str, default='v_normed_alltimes.txt')  # file to read data from (v_normed_alltimes.txt for normalized membrane potential or firings.txt for spike trains)
    parser.add_argument('--seeds', type=int, nargs="+", default=[10, 11, 12, 13, 14, 15, 16, 17, 18, 19])  # random seed for initializing model
    parser.add_argument('--train_ratio', type=float, default=0.6)  # fraction of data for training
    parser.add_argument('--val_ratio', type=float, default=0.2)  # fraction of data for validation
    parser.add_argument('--seq_len', type=int, default=10)  # history (aka context) length
    parser.add_argument('--loss', type=str, default='mse')  # loss function ('poisson' or 'bce' or 'mse')
    parser.add_argument('--prednstep', type=int, default=1)  # number of steps to predict
    parser.add_argument('--attn_enc_self_loc', type=str, default='full')  # encoder local self-attention type
    parser.add_argument('--attn_enc_self_glb', type=str, default='none')  # encoder global self-attention type
    parser.add_argument('--attn_dec_self_loc', type=str, default='none')  # decoder local self-attention type
    parser.add_argument('--attn_dec_self_glb', type=str, default='none')  # decoder global self-attention type
    parser.add_argument('--attn_dec_cross_loc', type=str, default='full')  # decoder local cross-attention type
    parser.add_argument('--attn_dec_cross_glb', type=str, default='full')  # decoder global cross-attention type
    parser.add_argument('--max_epoch', type=int, default=50)  # maximum number of epochs to run
    parser.add_argument('--device', type=str, default="cpu")  # "cpu", "mps", or "cuda:0"
    parser.add_argument('--patience', type=int, default=10)  # number of epochs to wait before early stopping
    parser.add_argument('--outdir', type=str, default='.')  # result directory
    parser.add_argument('--outname', type=str, default='best_model')   # output filename
    # model hyperparameter
    parser.add_argument('--d_model', type=int, default=100)  # model dimension
    parser.add_argument('--d_qkv', type=int, default=8)  # key, query, value dimension
    parser.add_argument('--n_heads', type=int, default=10)  # number of attention heads
    parser.add_argument('--ed_layers', type=int, default=1)  # number of encoder and decoder layers
    parser.add_argument('--time_emb_dim', type=int, default=1)  # time embedding dimension
    parser.add_argument('--dropout_emb', type=float, default=0.1)  # embedding dropout ratio
    parser.add_argument('--dropout_ff', type=float, default=0.1)  # feedforward network dropout ratio
    # optimizer hyperparameter
    parser.add_argument('--batch_size', type=int, default=16)  # batch size
    parser.add_argument('--base_lr', type=float, default=0.0005)  # base learning rate
    parser.add_argument('--l2_coeff', type=float, default=0.001)  # AdamW weight decay factor (L2 regularization)
    parser.add_argument('--warmup_steps', type=int, default=0)  # number of warmup steps for learning rate scheduler
    parser.add_argument('--decay_factor', type=float, default=0.5)  # learning rate decay factor
    args = parser.parse_args()

    device = torch.device(args.device)
    nloaders = 0
    if not os.path.exists(args.outdir):
        os.mkdir(args.outdir)

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2 ** 32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    ### load data ###
    l1 = args.expfolder.strip('/').split('/')
    explist = []
    for l in l1:
        explist.extend(l.split('_'))
    nexc = 0
    ninh = 0
    total_times = 0
    for e in explist:
        if e.startswith('exc'):
            nexc = int(e[3:])
        elif e.startswith('inh'):
            ninh = int(e[3:])
        elif e[0] == 't':
            e_ = e.split('/')
            total_times = int(e_[0][1:])
    print('nexc', nexc, 'ninh', ninh, 'total time', total_times)
    total_neuron = nexc + ninh

    assert total_times and total_neuron

    if args.datafile.startswith('firings'):
        with open(args.datafolder + args.expfolder + args.datafile) as f:
            lines = f.readlines()

        spikes_mat = np.zeros((total_times, total_neuron), dtype='float32') # total time steps x total number of neurons
        for l in lines:
            t, n = int(l.strip('\n').split(',')[0])-1, int(l.strip('\n').split(',')[1])-1  # matlab indices start at 1
            spikes_mat[t, n] = 1

    elif args.datafile.startswith('v'):
        V = np.loadtxt(args.datafolder + args.expfolder + args.datafile, delimiter=',', dtype='float32')
        spikes_mat = V.T

    else:
        print('data not found')

    x = np.linspace(0, total_times/1000, num=total_times, endpoint=False, dtype='float32')
    y = spikes_mat

    seq_len = args.seq_len
    prednstep = args.prednstep
    train_ratio = args.train_ratio
    val_ratio = args.val_ratio
    test_ratio = 1-args.train_ratio-args.val_ratio
    X_c = []
    Y_c = []
    X_t = []
    Y_t = []
    for i in range(len(x) - seq_len - prednstep + 1):
        X_c.append(x[i:i + seq_len])
        Y_c.append(y[i:i + seq_len, :])
        X_t.append(x[i + seq_len:i + seq_len + prednstep])
        Y_t.append(y[i + seq_len:i + seq_len + prednstep, :])
    X_c = np.array(X_c)
    Y_c = np.array(Y_c)
    X_t = np.array(X_t)
    if len(X_t.shape) == 1:
        X_t = X_t.reshape((-1, 1))
    Y_t = np.array(Y_t)
    if len(Y_t.shape) == 2:
        Y_t = np.expand_dims(Y_t, axis=1)
    X_c = np.expand_dims(X_c, axis=-1)
    X_t = np.expand_dims(X_t, axis=-1)
    print(X_c.shape, Y_c.shape, X_t.shape, Y_t.shape)
    total_samples = len(X_c)
    print('total samples', total_samples)
    X_c_train = X_c[0:int(total_samples * train_ratio)]
    Y_c_train = Y_c[0:int(total_samples * train_ratio)]
    X_t_train = X_t[0:int(total_samples * train_ratio)]
    Y_t_train = Y_t[0:int(total_samples * train_ratio)]
    X_c_val = X_c[int(total_samples * train_ratio):int(total_samples * (train_ratio + val_ratio))]
    Y_c_val = Y_c[int(total_samples * train_ratio):int(total_samples * (train_ratio + val_ratio))]
    X_t_val = X_t[int(total_samples * train_ratio):int(total_samples * (train_ratio + val_ratio))]
    Y_t_val = Y_t[int(total_samples * train_ratio):int(total_samples * (train_ratio + val_ratio))]
    X_c_test = X_c[int(total_samples * (train_ratio + val_ratio)):]
    Y_c_test = Y_c[int(total_samples * (train_ratio + val_ratio)):]
    X_t_test = X_t[int(total_samples * (train_ratio + val_ratio)):]
    Y_t_test = Y_t[int(total_samples * (train_ratio + val_ratio)):]
    print('training samples:', len(X_c_train), 'validation samples:', len(X_c_val), 'testing samples:', len(X_c_test))


    def train(dataloader, model, nlog):
        model.train()  # turn on train mode
        loss_all_batches = []

        for X_c_batch, Y_c_batch, X_t_batch, Y_t_batch in dataloader:  # batch_size, seq_len, n_var
            pred = model(X_c_batch, Y_c_batch, X_t_batch, Y_t_batch)[0]
            loss = criterion(pred, Y_t_batch.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_all_batches.append(loss.item())

        return np.sum(loss_all_batches[-nlog:]) / nlog


    def evaluate(dataloader, model):
        model.eval()  # turn on evaluation mode
        total_loss = 0.
        with torch.no_grad():
            for X_c_batch, Y_c_batch, X_t_batch, Y_t_batch in dataloader:
                pred = model(X_c_batch, Y_c_batch, X_t_batch, Y_t_batch)[0]
                total_loss += criterion(pred, Y_t_batch.to(device)).item()

        return total_loss / len(dataloader)

    for s in args.seeds:
        outname = args.outname+'_seed{}'.format(s)
        g = torch.Generator()
        g.manual_seed(s)
        torch.manual_seed(s)
        random.seed(s)
        np.random.seed(s)
        torch.cuda.manual_seed(s)
        torch.cuda.manual_seed_all(s)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = False

        train_loader = DataLoader(list(zip(X_c_train, Y_c_train, X_t_train, Y_t_train)), shuffle=True, batch_size=args.batch_size,
                                    num_workers=nloaders, worker_init_fn=seed_worker, generator=g)
        if val_ratio > 0:
            val_loader = DataLoader(list(zip(X_c_val, Y_c_val, X_t_val, Y_t_val)), shuffle=True, batch_size=args.batch_size,
                                        num_workers=nloaders, worker_init_fn=seed_worker, generator=g)
        if test_ratio > 0:
            test_loader = DataLoader(list(zip(X_c_test, Y_c_test, X_t_test, Y_t_test)), shuffle=True, batch_size=args.batch_size,
                                        num_workers=nloaders, worker_init_fn=seed_worker, generator=g)

        # initialize model
        if args.loss.lower() == 'poisson':  # negative poisson log likelihood
            criterion = nn.PoissonNLLLoss(log_input=False, reduction='mean')
            finalact = nn.Softplus()
        elif args.loss.lower() == 'bce':  # binary cross entropy
            criterion = nn.BCELoss()
            finalact = nn.Sigmoid()
        else:  # mean squared error
            criterion = nn.MSELoss()
            finalact = None

        CausalformerModel = Causalformer(d_yc=y.shape[-1], d_yt=y.shape[-1], d_x=1,
                                                d_model=args.d_model,
                                                d_queries_keys=args.d_qkv,
                                                d_values=args.d_qkv,
                                                n_heads=args.n_heads,
                                                e_layers=args.ed_layers,
                                                d_layers=args.ed_layers,
                                                time_emb_dim=args.time_emb_dim,
                                                dropout_emb=args.dropout_emb,
                                                dropout_ff=args.dropout_ff,
                                                finalact=finalact,
                                                d_ff=None,
                                                device=device,
                                                enc_global_self_attn=args.attn_enc_self_glb,
                                                enc_local_self_attn=args.attn_enc_self_loc,
                                                dec_global_cross_attn=args.attn_dec_cross_glb,
                                                dec_local_cross_attn=args.attn_dec_cross_loc,
                                                dec_global_self_attn=args.attn_dec_self_glb,
                                                dec_local_self_attn=args.attn_dec_self_loc)
        CausalformerModel.to(device)

        optimizer = torch.optim.AdamW(CausalformerModel.parameters(), lr=args.base_lr, weight_decay=args.l2_coeff)
        scheduler = WarmupReduceLROnPlateauScheduler(optimizer,
                                                        init_lr=1e-10,
                                                        peak_lr=args.base_lr,
                                                        warmup_steps=args.warmup_steps,
                                                        patience=3,
                                                        factor=args.decay_factor
        )

        train_hist = []
        val_hist = []
        best_val_loss = float('inf')
        saved_epoch = 0
        progfile = open(args.outdir+'/{}_prog.txt'.format(outname), 'w')
        hyperparams = []
        for arg in vars(args):
            if arg == 'seeds':
                hyperparams.append('seed: {} '.format(s))
            else:
                hyperparams.append('{}: {} '.format(arg, getattr(args, arg) or ''))
        hyperparams.append('\n')
        progfile.writelines(hyperparams)
        progfile.write('epoch    train    val    elapsed_sec\n')

        with TemporaryDirectory() as tempdir:
            best_model_params_path = os.path.join(tempdir, "best_model_params.pt")
            cnt = 0
            for epoch in range(1, args.max_epoch + 1):
                epoch_start_time = time.time()
                if val_ratio > 0:
                    train_loss_epoch = train(train_loader, CausalformerModel, len(val_loader))
                else:
                    train_loss_epoch = train(train_loader, CausalformerModel, len(train_loader))
                train_hist.append(train_loss_epoch)
                elapsed = time.time() - epoch_start_time
                if val_ratio == 0:
                    progfile.write('{}    {}    {}\n'.format(epoch, train_loss_epoch, elapsed))
                    scheduler.step(train_loss_epoch)
                else:
                    val_loss = evaluate(val_loader, CausalformerModel)
                    val_hist.append(val_loss)
                    progfile.write('{}    {}    {}    {}\n'.format(epoch, train_loss_epoch, val_loss, elapsed))
                    if np.isfinite(val_loss):
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            torch.save(CausalformerModel.state_dict(), best_model_params_path)
                            saved_epoch = epoch
                            cnt = 0
                        else:
                            cnt += 1
                            if cnt == args.patience:
                                progfile.write('early stopped at epoch {} \n'.format(epoch))
                                break
                    else:
                        progfile.write('diverged at epoch {} \n'.format(epoch))
                        break
                    scheduler.step(val_loss)
                # print(scheduler.get_lr())

            if val_ratio > 0:
                CausalformerModel.load_state_dict(torch.load(best_model_params_path))  # load best model states

            np.savez(args.outdir+'/{}_trainValHist.npz'.format(outname), train_hist=train_hist, val_hist=val_hist)
            torch.save(CausalformerModel.state_dict(), args.outdir+'/{}_bestModel.pth'.format(outname))

        if val_ratio == 0:
            progfile.write('best result at epoch {} train {} \n'.format(saved_epoch, train_hist[saved_epoch-1]))
        else:
            progfile.write('best result at epoch {} train {} val {} \n'.format(saved_epoch, train_hist[saved_epoch-1], val_hist[saved_epoch-1]))
        progfile.close()