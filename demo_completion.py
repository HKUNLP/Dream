import time
import torch
from transformers import AutoModel, AutoTokenizer

model_path = "Dream-org/Dream-v0-Instruct-7B"
model = AutoModel.from_pretrained(model_path, torch_dtype=torch.bfloat16, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = model.to("cuda").eval()

# messages = [
#     {"role": "user", "content": "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"},
# ]
messages = [
    {"role": "user", "content": "Please write a Python class that implements a PyTorch trainer capable of training a model on a toy dataset."}
]
inputs = tokenizer.apply_chat_template(
    messages, return_tensors="pt", return_dict=True, add_generation_prompt=True
)
input_ids = inputs.input_ids.to(device="cuda")
attention_mask = inputs.attention_mask.to(device="cuda")

output = model.diffusion_generate(
    input_ids,
    attention_mask=attention_mask,
    max_new_tokens=512,
    output_history=True,
    return_dict_in_generate=True,
    steps=512,
    temperature=0.2,
    top_p=0.95,
    alg="entropy",
    alg_temp=0.,
)
generations = [
    tokenizer.decode(g[len(p) :].tolist())
    for p, g in zip(input_ids, output.sequences)
]

print(generations[0].split(tokenizer.eos_token)[0])

# the following lines print the history of generations
history = output.history
for i, h in enumerate(history):
    print(f"############ Step {i} ############")
    time.sleep(0.01)
    # print(tokenizer.decode(h[0].tolist()).split(tokenizer.eos_token)[0].replace(tokenizer.mask_token, " "), end="\r")
    print(tokenizer.decode(h[0].tolist()), end="\r")



'''An example generation (maybe different due to randomness)
<|im_start|>system
You are a helpful assistant.<|im_end|>                                                                                                        <|im_start|>user    
Please write a Python class that implements a PyTorch trainer capable of training a model on a toy dataset.<|im_end|>
<|im_start|>assistant
Here's an example of a PyTorch trainer class that trains a model on a toy dataset:
                                   
```python                                                              
import torch                 
import torch.nn as nn                                                  
import torch.optim as optim
                                   
class Trainer:
    def __init__(self, model, optimizer, criterion):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
                                   
    def train(self, inputs, labels):              
        optimizer.zero_grad()
        outputs = self.model(inputs)
        loss = self.criterion(outputs, labels)                                                                                                
        loss.backward()
        optimizer.step()
        return loss.item()                                             
                                                                       
    def evaluate(self, inputs, labels):
        self.model.eval()                                              
        outputs = self.model(inputs)
        loss = self.criterion(outputs, labels)
        return loss.item()
        def predict(self, inputs, labels):                                 
        self.model.eval()
        outputs = self.model(inputs)    
        return outputs                                                 

    def save(self, path):                                              
        torch.save(self.model.state_dict(), path)                      
                                   
    def load(self, path):          
        self.model.load_state_dict(torch.load(path))                                                                                          
```                                                                                                                                           
                                   
To use this trainer class, you need to define a model, optimizer, and criterion, and then call the `train` method to train the model on your
dataset. For example:
                                                                                                                                              
```python            
class ToyModel(nn.Module):                                                                                                                    
    def __init__(self):            
        super(ToyModel, self).__init__()                               
        self = nn.Sequential(
            nn.Linear(1, nn.Linear(1, 1)),                             
            nn.ReLU(),     
            nn.Linear(1, 1)        
        )     
                                                                       
    def forward(self, x): 
        return self(x)            
                                   
model = ToyModel()                 
optimizer = optim.SGD(model.parameters(), lr=0.01)
criterion = nn.MSELoss()     
                                                                       
trainer = Trainer(model=model, optimizer=optimizer, criterion=criterion)                                                                      
                                   
for epoch in range(10): 
    loss = trainer.train(inputs, labels)                               
    print(f"Epoch {epoch+1}, Loss: {loss}")                            
                                                                       
test_loss = trainer.test(inputs, labels)                               
print(f"Test Loss: {test_loss}")                                       
```                                                                    
                                   
This example trains a simple neural network with one hidden layer on a toy dataset. The `train` method trains the model for each epoch, and
the `test` method evaluates the model's performance on unseen data. The `predict` method uses the trained model to make predictions.
'''
