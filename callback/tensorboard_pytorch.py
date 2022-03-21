# _*_coding:utf-8_*_
# 作者      ：46925
# 创建时间  ：2020/11/1615:01  
# 文件      ：tensorboard_pytorch.py
# IDE       ：PyCharm
from tensorboardX import SummaryWriter, FileWriter
from torchsummary import summary


def net_board(log_dir, comment_name, net, input_feature):
    with SummaryWriter(log_dir=log_dir, comment=comment_name) as write:
        write.add_graph(net, input_feature)
    print(summary(net, input_feature.shape, device="cpu"))


def loss_board(log_dir, comment_name, board_name, train_loss, valid_loss, epoch):
    with SummaryWriter(log_dir=log_dir, comment=comment_name) as write:
        write.add_scalars(board_name, {'train_loss':train_loss,
                                       'valid_loss':valid_loss}, epoch)


