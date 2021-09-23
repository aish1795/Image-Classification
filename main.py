#!/usr/bin/env python
# coding: utf-8

# # Dataloader

# In[ ]:


import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms


batch_size = 50
momentum = 0.9
lr = 0.01
epochs = 20
log_interval = 10

train_transform = transforms.Compose([
    transforms.ToPILImage(),
    # transforms.RandomVerticalFlip(p=0.5),
    # transforms.RandomRotation(degrees=(-30, 30)),
    transforms.RandomAffine(degrees=30,translate=(0.1, 0.2)),
    # transforms.RandomCrop(32, padding=4),
    # transforms.ColorJitter(brightness=0.1, contrast=0.2, saturation=0, hue=0),
    transforms.ToTensor(),
    # transforms.Normalize([0, 0, 0], [1, 1, 1])
])


class MyDataset(Dataset):

    def __init__(self, X_path="X.pt", y_path="y.pt", transforms=None):

        self.X = torch.load(X_path).squeeze(1)
        self.y = torch.load(y_path).squeeze(1)
        self.transforms = transforms
    
    def __len__(self):
        if self.transforms is not None:
          return 2*self.X.size(0)
        else:
          return self.X.size(0)

    def __getitem__(self, idx):
        if idx < (self.X.size(0)):
          image=self.X[idx]
        else:
          idx=idx%(self.X.size(0))
          image=self.X[idx]
          if self.transforms is not None:
              image = self.transforms(self.X[idx])
        return image, self.y[idx]
train_dataset = MyDataset(X_path="train/X.pt", y_path="train/y.pt", transforms=train_transform)
val_dataset = MyDataset(X_path="validation/X.pt", y_path="validation/y.pt", transforms=None)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)

val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=batch_size, shuffle=True, num_workers=1)


# In[ ]:


get_ipython().system(' npm install -g localtunnel')
get_ipython().system_raw('python3 -m pip install visdom')
get_ipython().system_raw('python3 -m visdom.server -port 6006 >> visdomlog.txt 2>&1 &')
get_ipython().system_raw('lt --port 6006 >> url.txt 2>&1 &')
import visdom


# In[ ]:


# print(len(train_dataset))
count=0
  print(data.shape)
for data, target in train_loader:
  print(data.shape)
  weights=data.detach()
  mins = weights.min(dim=1, keepdim=True)
  maxs = weights.max(dim=1, keepdim=True)
  tensor = (weights - mins[0]) / (maxs[0] - mins[0])
  tensor=tensor.reshape(50,3,32,32)
  vis = visdom.Visdom(port='6006')
  vis.images(weights)
  break


# # Model

# In[ ]:


import torch
import torch.nn as nn
import torch.nn.functional as F

nclasses = 43 # GTSRB has 43 classes

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layer1=nn.Sequential(
            nn.Conv2d(3, 32, 5),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(3,2),
            nn.Conv2d(32, 32, 3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(3,2)
        )
        self.layer2=nn.Sequential(
            nn.Conv2d(32, 64, 3, padding='same'),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding='same'),
            nn.ReLU()
        )
        self.layer3=nn.Sequential(
            nn.Conv2d(128, 256, 3, padding='same'),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.liner=nn.Sequential(
            nn.Linear(5*5*256, 500),
            nn.Dropout(0.5, True),
            nn.ReLU(),
            nn.Linear(500, 100),
            nn.ReLU()

        )
        self.out = nn.Linear(100, nclasses)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        # x=F.pad(x, (1, 1, 1, 1))  # [left, right, top, bot]
        x = self.layer3(x)
        # print(x.shape)
        x = torch.flatten(x, 1)
        x = self.liner(x)
        x = self.out(x)
        return F.log_softmax(x,dim=1)


# # Training

# In[ ]:


model = Net()
model.cuda()
lr = 0.01
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
scheduler=optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, patience=20)
# optimizer = optim.Adam(model.parameters(), lr=0.01, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
epochs = 500
best_loss=None
counter=0
min_loss=0.005
patience=7
early_stop=False
train_loss=[]
val_loss=[]
def train(epoch):
    runningLoss = 0
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data=data.cuda()
        target=target.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        runningLoss += loss.item()
    avg_loss = runningLoss/len(train_loader)
    print('Train Epoch: {} \tLoss: {:.6f}'.format(
        epoch, avg_loss))
    # train_loss.append(avg_loss) 

def validation():
    model.eval()
    validation_loss = 0
    correct = 0
    for data, target in val_loader:
        data=data.cuda()
        target=target.cuda()
        output = model(data)
        validation_loss += F.nll_loss(output, target, reduction="sum").item() # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    validation_loss /= len(val_loader.dataset)
    scheduler.step(validation_loss)
    print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        validation_loss, correct, len(val_loader.dataset),
        100. * correct / len(val_loader.dataset)))
    # val_loss.append(validation_loss)
    return validation_loss
  


for epoch in range(1, epochs + 1):
    train(epoch)
    validation_loss = validation()
    model_file = 'model_' + str(epoch) + '.pth'
    torch.save(model.state_dict(), model_file)
    print('\nSaved model to ' + model_file + '.')
    if best_loss == None:
            best_loss = validation_loss
    elif best_loss - validation_loss > min_loss:
        best_loss = validation_loss
        counter=0
    elif (best_loss - validation_loss) < min_loss:
        counter += 1
        # print(f"INFO: Early stopping counter {counter} of {patience}")
        if counter >= patience:
            # print('INFO: Early stopping')
            early_stop = True
    # if early_stop:
    #   break


# In[ ]:


import matplotlib.pyplot as plt

plt.figure(figsize=(10,5))
plt.title("Training and Validation Loss")
plt.plot(val_loss,label="val")
plt.plot(train_loss,label="train")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()


# # Evaluate
# 
# 

# In[ ]:




import pickle
import pandas as pd

outfile = 'gtsrb_kaggle.csv'

output_file = open(outfile, "w")
dataframe_dict = {"Filename" : [], "ClassId": []}

test_data = torch.load('testing/test.pt')
file_ids = pickle.load(open('testing/file_ids.pkl', 'rb'))
model = Net()
model.load_state_dict(torch.load('model_205.pth', map_location=torch.device('cpu')))
model.eval()
# weights=test_data.reshape(12630, 3 * 32 *32)
# mins = weights.min(dim=1, keepdim=True)
# maxs = weights.max(dim=1, keepdim=True)
# tensor = (weights - mins[0]) / (maxs[0] - mins[0])
# tensor=tensor.reshape(12630, 3,32,32)
for i, data in enumerate(test_data):
    data = data.unsqueeze(0)

    output = model(data)
    pred = output.data.max(1, keepdim=True)[1].item()
    file_id = file_ids[i][0:5]
    dataframe_dict['Filename'].append(file_id)
    dataframe_dict['ClassId'].append(pred)

df = pd.DataFrame(data=dataframe_dict)
df.to_csv(outfile, index=False)
print("Written to csv file {}".format(outfile))


# # Accuracy

# In[ ]:


import pandas as pd
gt = pd.read_csv("/content/drive/MyDrive/dataset/Test.csv")
gt['Path'] = gt['Path'].str[-9:-4]
gtf= pd.DataFrame()
gtf['x'] = gt['Path']
gtf['y'] = gt['ClassId']
gtf


# In[ ]:


ctr = 0
misclass = []
df=df.sort_values('Filename')
print(df['ClassId'])
for idx, i in enumerate(df['ClassId']):
  if(i == gtf['y'][idx]):
    ctr += 1
  else:
    misclass.append(idx)
print(ctr/len(gtf['y'])*100)

