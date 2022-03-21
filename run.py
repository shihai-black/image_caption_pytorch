import argparse
import json
import os.path
import time
from tqdm import tqdm

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision.transforms import transforms
from torch.optim import Adam, RMSprop
from torch.utils.data import Dataset, DataLoader
from nltk.translate.bleu_score import corpus_bleu

from configs import INPUT_ABSOLUTE_PATH
from utils.Logginger import init_logger
from utils.utils import adjust_learning_rate, AverageMeter, accuracy, save_checkpoints, load_checkpoints
from dataload import CaptionDataset
from models.models import Encoder, DecoderWithAttention
from callback.tensorboard_pytorch import net_board, loss_board

logger = init_logger('image2seq', './outputs/logs/')



def arguments():
    parser = argparse.ArgumentParser(description='image2seq Example')
    parser.add_argument('-bs', '--batch_size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 10000)')
    parser.add_argument('--train_grad_clip', type=int, default=5, metavar='N',
                        help='train grad clip (default: 5)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 5)')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='enables CUDA training(default: False)')
    parser.add_argument('--seed', type=int, default=1111, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--load', action='store_true', default=False,
                        help='Whether or not to load model(default: False)')
    parser.add_argument('-p', '--predict', action='store_true', default=False,
                        help='Predict or train(default: False)')
    parser.add_argument('-ft', '--fine_tune', action='store_true', default=False,
                        help='Whether or not to fine tune pre-model(default: False)')
    parser.add_argument('--log-interval', type=int, default=1024, metavar='N',
                        help='how many batches to wait before logging training status(default: 1024)')
    return parser.parse_args()


def load_word_map(data_name):
    word_map_file = os.path.join(INPUT_ABSOLUTE_PATH, data_name, 'word_map.json')
    with open(word_map_file, 'r') as f:
        word_map = json.load(f)
    return word_map


def train(train_loader, encoder, decoder, loss_func, encoder_opt, decoder_opt, epoch, device):
    encoder.train()
    decoder.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top5accs = AverageMeter()

    start = time.time()

    for i, (imgs, caps, caplens) in enumerate(train_loader):
        data_time.update(time.time() - start)

        imgs = imgs.to(device)
        caps = caps.to(device)
        caplens = caplens.to(device)

        # 前向传播
        imgs = encoder(imgs)
        scores, caps_sorted, decode_lengths, alphs, sort_ind = decoder(imgs, caps, caplens)
        targets = caps_sorted[:, 1:]

        scores = pack_padded_sequence(scores, decode_lengths, batch_first=True).data
        targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data

        loss = loss_func(scores, targets)
        loss += ((1. - alphs.sum(dim=1)) ** 2).mean()

        decoder_opt.zero_grad()
        if encoder_opt is not None:
            encoder_opt.zero_grad()
        loss.backward()

        # 梯度裁剪
        nn.utils.clip_grad_norm_(decoder.parameters(), args.train_grad_clip)
        if encoder_opt is not None:
            nn.utils.clip_grad_norm_(encoder.parameters(), args.train_grad_clip)

        # 更新参数
        decoder_opt.step()
        if encoder_opt is not None:
            encoder_opt.step()

        # metrics
        top5 = accuracy(scores, targets, 5)
        losses.update(loss.item(), sum(decode_lengths))
        top5accs.update(top5, sum(decode_lengths))
        batch_time.update(time.time() - start)

        # Print status
        if i % args.batch_size == 0:
            logger.info('Epoch: [{0}][{1}/{2}]\t'
                        'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})'.format(epoch, i, len(train_loader),
                                                                                batch_time=batch_time,
                                                                                data_time=data_time, loss=losses,
                                                                                top5=top5accs))
    return losses.avg


@torch.no_grad()  # 停止autograd模块，停止梯度计算
def valid(word_map, val_loader, encoder, decoder, loss_func, device):
    decoder.eval()  # 切换dropout层和batchnorm层
    if encoder is not None:
        encoder.eval()
    batch_time = AverageMeter()
    losses = AverageMeter()
    top5accs = AverageMeter()

    start = time.time()
    references = list()  # references (true captions) for calculating BLEU-4 score
    hypotheses = list()  # hypotheses (predictions)

    for i, (imgs, caps, caplens, allcaps) in enumerate(val_loader):
        # Move to device, if available
        imgs = imgs.to(device)
        caps = caps.to(device)
        caplens = caplens.to(device)

        if encoder is not None:
            imgs = encoder(imgs)
        scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs, caps, caplens)

        targets = caps_sorted[:, 1:]

        # Remove timesteps that we didn't decode at, or are pads
        # pack_padded_sequence is an easy trick to do this
        scores_copy = scores.clone()
        scores = pack_padded_sequence(scores, decode_lengths, batch_first=True).data
        targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data

        loss = loss_func(scores, targets)

        loss += (1 - alphas.sum(dim=1) ** 2).mean()

        # 更新评价指标
        losses.update(loss.item(), sum(decode_lengths))
        top5 = accuracy(scores, targets, 5)
        top5accs.update(top5, sum(decode_lengths))
        batch_time.update(time.time() - start)

        start = time.time()

        if i % args.batch_size == 0:
            logger.info('Validation: [{0}/{1}]\t'
                        'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})\t'.format(i, len(val_loader),
                                                                                  batch_time=batch_time,
                                                                                  loss=losses, top5=top5accs))
        # References
        allcaps = allcaps[sort_ind]  # because images were sorted in the decoder
        for j in range(allcaps.shape[0]):
            img_caps = allcaps[j].tolist()
            img_captions = list(
                map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<pad>']}],
                    img_caps))  # remove <start> and pads
            references.append(img_captions)

        # Hypotheses
        _, preds = torch.max(scores_copy, dim=2)
        preds = preds.tolist()
        temp_preds = list()
        for j, p in enumerate(preds):
            temp_preds.append(preds[j][:decode_lengths[j]])  # remove pads
        preds = temp_preds
        hypotheses.extend(preds)

        assert len(references) == len(hypotheses)

        # Calculate BLEU-4 scores
    bleu4 = corpus_bleu(references, hypotheses)

    logger.info('\n * LOSS - {loss.avg:.3f}, TOP-5 ACCURACY - {top5.avg:.3f}, BLEU-4 - {bleu}\n'.format(
        loss=losses,
        top5=top5accs,
        bleu=bleu4))

    return bleu4, losses.avg


@torch.no_grad()
def evaluate(word_map, test_loader, encoder, decoder, beam_size, device):
    vocab_size = len(word_map)
    if encoder is not None:
        encoder.eval()
    decoder.eval()
    references = list()
    hypotheses = list()
    for i, (image, caps, caplens, allcaps) in enumerate(
            tqdm(test_loader, desc="EVALUATING AT BEAM SIZE " + str(beam_size))):
        k = beam_size

        image = image.to(device)
        encoder_out = encoder(image)
        encoder_dim = encoder_out.size(3)

        encoder_out = encoder_out.view(1, -1, encoder_dim)  # (1, num_pixels, encoder_dim)
        num_pixels = encoder_out.size(1)

        encoder_out = encoder_out.expand(k, num_pixels, encoder_dim)  # (k, num_pixels, encoder_dim)

        k_prev_words = torch.LongTensor([[word_map['<start>']]] * k).to(device)  # (k, 1)

        seqs = k_prev_words  # (k, 1)

        complete_seqs = list()
        complete_seqs_scores = list()

        # Start decoding
        step = 1
        h, c = decoder.init_hidden_state(encoder_out)

        while True:
            embeddings = decoder.embedding(k_prev_words).squeeze(1)  # (s, embed_dim)

            awe, _ = decoder.attention(encoder_out, h)  # (s, encoder_dim), (s, num_pixels)

            gate = decoder.sigmoid(decoder.f_beta(h))  # gating scalar, (s, encoder_dim)

            awe = gate * awe

            h, c = decoder.decode_step(torch.cat([embeddings, awe], dim=1), (h, c))  # (s, decoder_dim)

            scores = decoder.fc(h)

            scores = F.log_softmax(scores, dim=1)
            top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)

            # prev_word_inds = top_k_words / vocab_size  # (s)
            next_word_inds = top_k_words % vocab_size  # (s)

            # add new words to sequences
            # prev_word_inds = prev_word_inds.type(torch.long)

            seqs = torch.cat([seqs, next_word_inds.unsqueeze(1)], dim=1)

            # Which sequences are incomplete (didn't reach <end>)?
            incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                               next_word != word_map['<end>']]
            complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

            # Set aside complete sequences
            if len(complete_inds) > 0:
                complete_seqs.extend(seqs[complete_inds].tolist())
                complete_seqs_scores.extend(top_k_scores[complete_inds])
            k -= len(complete_inds)  # reduce beam length accordingly

            # Proceed with incomplete sequences
            if k == 0:
                break
            k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)
            # Break if things have been going on too long
            if step > 50:
                break
            step += 1
        seq = seqs.tolist()[0]

        # References
        img_caps = allcaps[0].tolist()
        img_captions = list(
            map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}],
                img_caps))  # remove <start> and pads
        references.append(img_captions)

        # Hypotheses
        hypothese = [w for w in seq if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}]
        hypotheses.append(hypothese)

        assert len(references) == len(hypotheses)

    # Calculate BLEU-4 scores
    all_bleu4 = corpus_bleu(references, hypotheses)
    print(all_bleu4)


def cmd_entry(args):
    # 局部参数
    atten_dim = 512
    embed_dim = 512
    decoder_dim = 512
    dropout = 0.5
    best_bleu4 = 0
    epochs_since_improvement = 0
    checkpoint = None
    encoder_lr = 1e-4  # learning rate for encoder if fine-tuning
    decoder_lr = 4e-4  # learning rate for decoder
    start_epoch = 0
    fine_tune_encoder = False
    data_name = 'Flicker8k'
    word_map = load_word_map(data_name)
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    device = torch.device("cuda:1" if args.cuda else "cpu")

    # 载入数据
    logger.info('Loading data started ...')
    data_folder = os.path.join(INPUT_ABSOLUTE_PATH, data_name)
    normalize = transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                         std=[0.229, 0.224, 0.225])])
    train_dataset = CaptionDataset(data_folder, 'TRAIN', normalize)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1, pin_memory=True)

    val_dataset = CaptionDataset(data_folder, 'VAL', normalize)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1, pin_memory=True)

    test_dataset = CaptionDataset(data_folder, 'TEST', normalize)
    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=1, pin_memory=True)

    # 构建模型
    logger.info('Loading network started ...')
    encoder = Encoder().to(device)
    if args.fine_tune:  # 是否微调模型
        encoder.fine_tune(True)
        encoder_opt = RMSprop(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                           lr=encoder_lr)
    else:
        encoder.fine_tune(False)
        encoder_opt = None
    decoder = DecoderWithAttention(attention_dim=atten_dim,
                                   embed_dim=embed_dim,
                                   decoder_dim=decoder_dim,
                                   vocab_size=len(word_map),
                                   device=device,
                                   dropout=dropout).to(device)
    decoder_opt = RMSprop(params=filter(lambda p: p.requires_grad, decoder.parameters()),
                       lr=decoder_lr)

    # loss function
    loss_func = nn.CrossEntropyLoss().to(device)

    if args.predict:  # 预测
        checkpoint = load_checkpoints('./outputs/', data_name)
        decoder = checkpoint['decoder'].to(device)
        encoder = checkpoint['encoder'].to(device)
        evaluate(word_map, test_loader, encoder, decoder, 1, device)
    else:
        for epoch in range(start_epoch, args.epochs):
            if epochs_since_improvement == 5:
                break
            if epochs_since_improvement > 0 and epochs_since_improvement % 2 == 0:
                adjust_learning_rate(decoder_opt, 0.8)
                if fine_tune_encoder:
                    adjust_learning_rate(encoder_opt, 0.8)

            # train
            train_loss = train(train_loader, encoder, decoder, loss_func, encoder_opt, decoder_opt, epoch, device)

            # valid
            recent_bleu4, valid_loss = valid(word_map, val_loader, encoder, decoder, loss_func, device)

            # Check if there was an improvement
            is_best = recent_bleu4 > best_bleu4
            best_bleu4 = max(recent_bleu4, best_bleu4)
            if not is_best:
                epochs_since_improvement += 1
                logger.info("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
            else:
                epochs_since_improvement = 0
            # Save checkpoint
            save_checkpoints('./outputs/', data_name, epoch, epochs_since_improvement, encoder, decoder, encoder_opt,
                             decoder_opt, recent_bleu4, is_best)
            loss_board('./outputs/board', 'train', 'loss', train_loss, valid_loss, epoch)


if __name__ == '__main__':
    args = arguments()
    logger.info(f'Arguments:{args}')
    torch.manual_seed(args.seed)
    cmd_entry(args)
