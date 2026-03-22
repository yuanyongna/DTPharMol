import os
import json
import re
from transformers import AutoTokenizer, PreTrainedTokenizerFast


class Tokenizer_forRL:

    def __init__(self, args):
        if args.vocab == "bert":
            tokenizer = AutoTokenizer.from_pretrained(args.config_name)
            self.tokenizer = tokenizer
            self.sep_token_id = tokenizer.sep_token_id
            self.pad_token_id = tokenizer.pad_token_id
            tokenizer.save_pretrained(args.checkpoint_path)
        else:
            vocab_dict = {
                "[START]": 0,
                "[END]": 1,
                "[UNK]": 2,
                "[PAD]": 3,
                "[UNCONDITION]": 4,
                "[PROP]": 5,
            }
            with open(args.vocab, "r", encoding="utf-8") as f:
                for row in f:
                    vocab_dict[row.strip().split(" ")[0]] = len(vocab_dict)
            self.tokenizer = vocab_dict
            self.rev_tokenizer = {v: k for k, v in vocab_dict.items()}
            self.sep_token_id = vocab_dict["[END]"]
            self.pad_token_id = vocab_dict["[PAD]"]
            # 仅允许主进程保存完整的词汇文件, 以避免多个进程同时写入文件
            if int(os.environ["LOCAL_RANK"]) == 0:
                path_save_vocab = f"{args.checkpoint_path}/vocab.json"
                with open(path_save_vocab, "w") as f:
                    json.dump(vocab_dict, f)
        self.vocab_size = len(self.tokenizer)
        args.vocab_size = self.vocab_size
        self.mask = args.mask

    def encode_token(
        self, sentences, ppgraph_len=False, num_props=False, scaffold=False
    ):
        pattern = "(\[[^\]]+]|<|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
        regex = re.compile(pattern)

        if ppgraph_len and num_props and scaffold:
            input_ids = []
            for seq in sentences:
                tmp = []
                ppgraph = [float(i) for i in seq[0:ppgraph_len]]
                prop = [float(j) for j in seq[ppgraph_len : ppgraph_len + num_props]]
                scaf = seq[ppgraph_len + num_props]
                tmp += ppgraph
                tmp += prop
                tmp += [
                    self.tokenizer.get(x, self.tokenizer["[UNK]"])
                    for x in regex.findall(scaf.strip())
                ]
                input_ids.append(tmp)
                # print("input_ids:\n", input_ids)
        elif ppgraph_len and num_props:
            input_ids = []
            for seq in sentences:
                tmp = []
                ppgraph = [float(i) for i in seq[0:ppgraph_len]]
                prop = [float(j) for j in seq[ppgraph_len : ppgraph_len + num_props]]
                # scaf = seq[ppgraph_len+num_props]
                tmp += ppgraph
                tmp += prop
                # tmp+=[self.tokenizer.get(x, self.tokenizer["[UNK]"]) for x in regex.findall(scaf.strip())]
                input_ids.append(tmp)
                # print("input_ids:\n", input_ids)
        elif ppgraph_len and scaffold:
            input_ids = []
            for seq in sentences:
                tmp = []
                ppgraph = [float(i) for i in seq[0:ppgraph_len]]
                # prop = [float(j) for j in seq[ppgraph_len : ppgraph_len+num_props]]
                scaf = seq[ppgraph_len + num_props]
                tmp += ppgraph
                # tmp += prop
                # tmp+=[self.tokenizer.get(x, self.tokenizer["[UNK]"]) for x in regex.findall(scaf.strip())]
                input_ids.append(tmp)
                # print("input_ids:\n", input_ids)
        elif num_props and scaffold:
            input_ids = []
            for seq in sentences:
                tmp = []
                # ppgraph = [float(i) for i in seq[0 : ppgraph_len]]
                prop = [float(j) for j in seq[ppgraph_len : ppgraph_len + num_props]]
                scaf = seq[ppgraph_len + num_props]
                # tmp += ppgraph
                tmp += prop
                tmp += [
                    self.tokenizer.get(x, self.tokenizer["[UNK]"])
                    for x in regex.findall(scaf.strip())
                ]
                input_ids.append(tmp)
                # print("input_ids:\n", input_ids)
        elif ppgraph_len:
            input_ids = []
            for seq in sentences:
                tmp = []
                # print("seq: ", seq)
                ppgraph = [float(i) for i in seq[0:ppgraph_len]]
                # prop = [float(j) for j in seq[ppgraph_len : ppgraph_len+num_props]]
                # scaf = seq[ppgraph_len+num_props]
                tmp += ppgraph
                # tmp += prop
                # tmp+=[self.tokenizer.get(x, self.tokenizer["[UNK]"]) for x in regex.findall(scaf.strip())]
                input_ids.append(tmp)
                # print("input_ids:\n", input_ids)
        elif num_props:
            input_ids = []
            for seq in sentences:
                tmp = []
                # ppgraph = [float(i) for i in seq[0 : ppgraph_len]]
                prop = [float(j) for j in seq[ppgraph_len : ppgraph_len + num_props]]
                # scaf = seq[ppgraph_len+num_props]
                # tmp += ppgraph
                tmp += prop
                # tmp+=[self.tokenizer.get(x, self.tokenizer["[UNK]"]) for x in regex.findall(scaf.strip())]
                input_ids.append(tmp)
                # print("input_ids:\n", input_ids)
        elif scaffold:
            input_ids = []
            for seq in sentences:
                tmp = []
                # ppgraph = [float(i) for i in seq[0 : ppgraph_len]]
                # prop = [float(j) for j in seq[ppgraph_len : ppgraph_len+num_props]]
                scaf = seq[ppgraph_len + num_props]
                # tmp += ppgraph
                # tmp += prop
                tmp += [
                    self.tokenizer.get(x, self.tokenizer["[UNK]"])
                    for x in regex.findall(scaf.strip())
                ]
                input_ids.append(tmp)
                # print("input_ids:\n", input_ids)
        else:
            input_ids = [
                [0]
                + [
                    self.tokenizer.get(x, self.tokenizer["[UNK]"])
                    for x in regex.findall(seq.strip())
                ]
                + [1]
                for seq in sentences
            ]

        """
        if scaffold and num_props:
            input_ids=[]
            for seq in sentences:
                tmp=[]
                prop, scaf=[float(i) for i in seq[:num_props+ppgraph_len]], seq[num_props+ppgraph_len]
                tmp+=prop
                tmp+=[self.tokenizer.get(x, self.tokenizer["[UNK]"]) for x in regex.findall(scaf.strip())]
                input_ids.append(tmp)
                # print("input_ids:\n", input_ids)
        elif num_props:
            input_ids=[]
            for seq in sentences:
                tmp=[]
                if isinstance(seq, list):
                    tmp+=seq
                else:
                    tmp+=[self.tokenizer.get(x, self.tokenizer["[UNK]"]) for x in regex.findall(seq.strip())]
                input_ids.append(tmp)
        else:
            input_ids = [[0] + [self.tokenizer.get(x, self.tokenizer["[UNK]"]) for x in regex.findall(seq.strip())] + [1] for seq in sentences]
        """

        return input_ids

    def decode_token(self, seq):
        # logger.log("\n", "进入 decode_token 方法:\n", "seq:", seq.squeeze(-1).tolist())
        if isinstance(self.tokenizer, dict):
            seq = seq.squeeze(-1).tolist()
            while len(seq) > 0 and seq[-1] == self.pad_token_id:
                seq.pop()
            tokens = (
                " ".join([self.rev_tokenizer[x] for x in seq])
                .replace("__ ", "")
                .replace("@@ ", "")
            )
        elif isinstance(self.tokenizer, PreTrainedTokenizerFast):
            seq = seq.squeeze(-1).tolist()
            while len(seq) > 0 and seq[-1] == self.pad_token_id:
                seq.pop()
            tokens = self.tokenizer.decode(seq)
        else:
            assert False, "invalid type of vocab_dict"
        return tokens
