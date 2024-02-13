import torch
import pandas as pd
import numpy as np
import torch
import torch.optim as optim
from transformers import AutoTokenizer, AutoModelWithLMHead
import json
from sklearn.metrics import mean_squared_error


class GPT2Movie(torch.nn.Module):
    def __init__(self, device: str, pretrained_model: str=None):
        super().__init__()
        self.model = None
        self.device = device
        if pretrained_model:
            self.model = AutoModelWithLMHead.from_pretrained(pretrained_model)
        else:
            self.model = AutoModelWithLMHead.from_pretrained("gpt2")
        self.model = self.model.to(self.device)
        self.tokenizer = tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.optimizer = optim.AdamW(self.model.parameters(), lr=5e-4)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.model(tensor)

    def train(self,train_dataloader, epochs: int) -> None:
        for epoch in range(epochs):
            total_loss = 0.0
            for idx, batch in enumerate(train_dataloader):
                with torch.set_grad_enabled(True):
                    self.optimizer.zero_grad()
                    batch = batch.to(self.device)
                    output = self.model(batch, labels=batch)
                    loss = output[0]
                    loss.backward()
                    self.optimizer.step()
                    # if idx % 100 == 0:
                    #     print("loss: %f, %d"%(loss, idx))
                    total_loss += loss.item()
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(train_dataloader)}")


    def save(self,filepath: str="model/") -> None:
        self.model.save_pretrained(save_directory=filepath)
        self.tokenizer.save_vocabulary(save_directory=filepath)
        
    def topk(self,probs: torch.Tensor, k: int=5) -> int:
        probs = torch.softmax(probs, dim= -1)

        tokensProb, topIx = torch.topk(probs, k=k)
        tokensProb = tokensProb / torch.sum(tokensProb)
        tokensProb = tokensProb.cpu().detach().numpy()

        choice = np.random.choice(k, 1, p = tokensProb)
        tokenId = topIx[choice][0]

        return int(tokenId)
    
    def inference(self, init_token: torch.Tensor, max_length: int=10) -> str:

        sequence = init_token.numpy().tolist()
        init_input = init_token.unsqueeze(0).to(self.device)

        with torch.set_grad_enabled(False):
            output = self.model(init_input)
            logits = output.logits[0,-1]

            sequence.append(self.topk(logits))

            for i in range(max_length):
                inp = torch.tensor(sequence).unsqueeze(0).to(self.device)
                output = self.model(inp)
                logits = output.logits[0,-1]
                res_id = self.topk(logits)

                if res_id == self.tokenizer.eos_token_id:
                    return self.tokenizer.decode(sequence)
                else: 
                    sequence.append(res_id)

        return self.tokenizer.decode(sequence)

    def eval(self,test_dataset,max_len:int=10) -> None:
        results = []
        within_max_len = 0
        within_req_len = 0
        equal_req_len = 0
        req_len = []
        gen_len = []
        for inp in test_dataset:
            ret_seq = self.inference(inp,max_len).strip()
            results.append(ret_seq)
            true_len = int(ret_seq.split("<text>")[0].split(" ")[1])
            output = ret_seq.split("<text>")[1].split(" ")[1:]
            # print(req_len,len(output),output)
            if len(output)<=max_len:
                within_max_len+=1
            if len(output)<=true_len:
                within_req_len+=1
                if len(output)==true_len:
                    equal_req_len+=1
            req_len.append(true_len)
            gen_len.append(len(output))
            
        
        result_json = {"within_max_len":within_max_len/len(test_dataset),
                        "within_req_len": within_req_len/len(test_dataset),
                        "equal_req_len":equal_req_len/len(test_dataset),
                        "MSE_genvreq":mean_squared_error(req_len,gen_len),
                        "gen_results":results}
                        
        json_file_path = "eval_results.json"

        with open(json_file_path, "w") as json_file:
            json.dump(result_json, json_file, indent=4)
        
        print(f"Output within max seq length: {within_max_len/len(test_dataset)}")
        print(f"Output within req seq length: {within_req_len/len(test_dataset)}")
        print(f"Output equal req seq length: {equal_req_len/len(test_dataset)}")
        print(f"MSE req vs gen seq length: {mean_squared_error(req_len,gen_len)}")
        print("-"*20)
        print(results[:10])
    