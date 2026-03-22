import torch
import os
import time


def create_embedding_model(args, vocab_size):
    model = torch.nn.Embedding(vocab_size, args.hidden_dim)
    path_save = "{}/my_random_emb.torch".format(args.checkpoint_path)
    path_save_ind = path_save + ".done"
    # 仅允许主进程初始化和保存嵌入模型
    if int(os.environ["LOCAL_RANK"]) == 0:
        if os.path.exists(path_save):
            model.load_state_dict(torch.load(path_save))
        else:
            torch.nn.init.normal_(model.weight)
            torch.save(model.state_dict(), path_save)
            os.sync()  # 确保所有文件系统的写入操作都已完成
            with open(
                path_save_ind, "x"
            ) as _:  # "x" 模式表示“独占创建”, 即仅在文件不存在时创建文件, 如果文件已存在, 则会抛出一个 FileExistsError
                pass
    # 对于其他进程, 进入一个循环, 直到找到指示文件 (.done) , 以确保嵌入已经初始化完成
    else:
        while not os.path.exists(path_save_ind):
            time.sleep(1)
        model.load_state_dict(torch.load(path_save))
    return model
