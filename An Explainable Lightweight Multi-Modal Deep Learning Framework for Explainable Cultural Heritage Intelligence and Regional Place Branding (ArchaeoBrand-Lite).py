# ==========================================================
# ArchaeoBrandLite - Part 1
# Install:
# pip install torch torchvision transformers pandas numpy
#             scikit-learn pillow shap
# ==========================================================

import os
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from transformers import DistilBertTokenizer
from transformers import DistilBertModel

from torchvision.models import mobilenet_v3_small
from torchvision import transforms

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------------------------------------
# Dataset
# CSV format:
# image,text,latitude,longitude,label
# ----------------------------------------------------------

CSV_FILE="dataset.csv"

df=pd.read_csv(CSV_FILE)

encoder=LabelEncoder()
df["label"]=encoder.fit_transform(df["label"])

train_df,test_df=train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)

# ----------------------------------------------------------
# Tokenizer
# ----------------------------------------------------------

tokenizer=DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

transform=transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

# ----------------------------------------------------------
# Dataset class
# ----------------------------------------------------------

class HeritageDataset(Dataset):

    def __init__(self,data):

        self.data=data.reset_index(drop=True)

    def __len__(self):

        return len(self.data)

    def __getitem__(self,index):

        row=self.data.iloc[index]

        img=Image.open(row.image).convert("RGB")
        img=transform(img)

        token=tokenizer(
            row.text,
            max_length=64,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        ids=token["input_ids"].squeeze(0)
        mask=token["attention_mask"].squeeze(0)

        geo=torch.tensor(
            [row.latitude,row.longitude],
            dtype=torch.float32
        )

        label=torch.tensor(
            row.label,
            dtype=torch.long
        )

        return img,ids,mask,geo,label

train_loader=DataLoader(
    HeritageDataset(train_df),
    batch_size=8,
    shuffle=True
)

test_loader=DataLoader(
    HeritageDataset(test_df),
    batch_size=8
)

# ----------------------------------------------------------
# DistilBERT
# ----------------------------------------------------------

class TextEncoder(nn.Module):

    def __init__(self):

        super().__init__()

        self.bert=DistilBertModel.from_pretrained(
            "distilbert-base-uncased"
        )

    def forward(self,ids,mask):

        x=self.bert(
            input_ids=ids,
            attention_mask=mask
        )

        return x.last_hidden_state[:,0]

# ----------------------------------------------------------
# MobileNetV3
# ----------------------------------------------------------

class ImageEncoder(nn.Module):

    def __init__(self):

        super().__init__()

        net=mobilenet_v3_small(weights="DEFAULT")

        self.features=net.features

        self.pool=nn.AdaptiveAvgPool2d(1)

    def forward(self,x):

        x=self.features(x)

        x=self.pool(x)

        x=x.flatten(1)

        return x

# ----------------------------------------------------------
# Spatial Encoder
# ----------------------------------------------------------

class SpatialEncoder(nn.Module):

    def __init__(self):

        super().__init__()

        self.net=nn.Sequential(

            nn.Linear(2,32),
            nn.ReLU(),

            nn.Linear(32,64),
            nn.ReLU()

        )

    def forward(self,x):

        return self.net(x)

# ----------------------------------------------------------
# Sparse Autoencoder
# ----------------------------------------------------------

class SparseAE(nn.Module):

    def __init__(self,input_dim,latent):

        super().__init__()

        self.encoder=nn.Sequential(

            nn.Linear(input_dim,512),
            nn.ReLU(),

            nn.Linear(512,latent)

        )

        self.decoder=nn.Sequential(

            nn.Linear(latent,512),
            nn.ReLU(),

            nn.Linear(512,input_dim)

        )

    def forward(self,x):

        z=self.encoder(x)

        r=self.decoder(z)

        return z,r

# ----------------------------------------------------------
# TinyMLP
# ----------------------------------------------------------

class TinyMLP(nn.Module):

    def __init__(self,input_dim,classes):

        super().__init__()

        self.net=nn.Sequential(

            nn.Linear(input_dim,256),
            nn.ReLU(),
            nn.Dropout(.3),

            nn.Linear(256,128),
            nn.ReLU(),

            nn.Linear(128,classes)

        )

    def forward(self,x):

        return self.net(x)

# ----------------------------------------------------------
# Complete Model
# ----------------------------------------------------------

class HeritageNet(nn.Module):

    def __init__(self,n_classes):

        super().__init__()

        self.text=TextEncoder()

        self.image=ImageEncoder()

        self.geo=SpatialEncoder()

        self.ae=SparseAE(
            576+768,
            256
        )

        self.cls=TinyMLP(
            256,
            n_classes
        )

    def forward(self,img,ids,mask,geo):

        t=self.text(ids,mask)

        i=self.image(img)

        g=self.geo(geo)

        fusion=torch.cat([t,i,g],1)

        latent,recon=self.ae(fusion)

        out=self.cls(latent)

        return out,recon,fusion

# ----------------------------------------------------------
# Train
# ----------------------------------------------------------

model=HeritageNet(
    len(encoder.classes_)
).to(device)

ce=nn.CrossEntropyLoss()
mse=nn.MSELoss()

opt=torch.optim.Adam(
    model.parameters(),
    lr=1e-4
)

for epoch in range(10):

    model.train()

    total=0

    for img,ids,mask,geo,label in train_loader:

        img=img.to(device)
        ids=ids.to(device)
        mask=mask.to(device)
        geo=geo.to(device)
        label=label.to(device)

        pred,recon,fusion=model(
            img,
            ids,
            mask,
            geo
        )

        loss=ce(pred,label)+0.05*mse(recon,fusion)

        opt.zero_grad()
        loss.backward()
        opt.step()

        total+=loss.item()

    print(epoch+1,total/len(train_loader))

# ----------------------------------------------------------
# Test
# ----------------------------------------------------------

model.eval()

correct=0
total=0

with torch.no_grad():

    for img,ids,mask,geo,label in test_loader:

        img=img.to(device)
        ids=ids.to(device)
        mask=mask.to(device)
        geo=geo.to(device)
        label=label.to(device)

        pred,_,_=model(
            img,
            ids,
            mask,
            geo
        )

        pred=pred.argmax(1)

        correct+=(pred==label).sum().item()

        total+=label.size(0)

print("Accuracy =",100*correct/total,"%")

torch.save(
    model.state_dict(),
    "ArchaeoBrandLite.pth"
)

print("Model Saved")
# ==========================================================
# ArchaeoBrandLite - Part 2
# Prediction + SHAP + Chatbot + Evaluation
# ==========================================================

import os
import torch
import numpy as np
import pandas as pd
from PIL import Image

from torchvision import transforms
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

import matplotlib.pyplot as plt
import shap

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------
# Load Model
# ---------------------------------------------------------

model=HeritageNet(
    len(encoder.classes_)
).to(device)

model.load_state_dict(
    torch.load(
        "ArchaeoBrandLite.pth",
        map_location=device
    )
)

model.eval()

# ---------------------------------------------------------
# Evaluation
# ---------------------------------------------------------

true=[]
predicted=[]

with torch.no_grad():

    for img,ids,mask,geo,label in test_loader:

        img=img.to(device)
        ids=ids.to(device)
        mask=mask.to(device)
        geo=geo.to(device)

        out,_,_=model(
            img,
            ids,
            mask,
            geo
        )

        p=out.argmax(1).cpu().numpy()

        predicted.extend(p)

        true.extend(label.numpy())

acc=accuracy_score(true,predicted)
pre=precision_score(true,predicted,average="weighted")
rec=recall_score(true,predicted,average="weighted")
f1=f1_score(true,predicted,average="weighted")

print("="*60)
print("Accuracy :",acc)
print("Precision:",pre)
print("Recall   :",rec)
print("F1 Score :",f1)
print("="*60)

print(classification_report(
    true,
    predicted,
    target_names=encoder.classes_
))

cm=confusion_matrix(true,predicted)

plt.figure(figsize=(7,7))
plt.imshow(cm)
plt.colorbar()
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

# ---------------------------------------------------------
# Single Prediction
# ---------------------------------------------------------

transform=transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

def predict(
    image_path,
    text,
    latitude,
    longitude
):

    image=Image.open(image_path).convert("RGB")

    image=transform(image).unsqueeze(0)

    token=tokenizer(
        text,
        max_length=64,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    )

    ids=token["input_ids"]
    mask=token["attention_mask"]

    geo=torch.tensor(
        [[latitude,longitude]],
        dtype=torch.float32
    )

    image=image.to(device)
    ids=ids.to(device)
    mask=mask.to(device)
    geo=geo.to(device)

    with torch.no_grad():

        out,_,_=model(
            image,
            ids,
            mask,
            geo
        )

        prob=torch.softmax(out,1)

        index=torch.argmax(prob).item()

    return (
        encoder.classes_[index],
        prob[0,index].item()
    )

# ---------------------------------------------------------
# Example
# ---------------------------------------------------------

label,score=predict(
    "sample.jpg",
    "Ancient temple with stone carvings",
    10.790,
    78.700
)

print("Prediction :",label)
print("Confidence :",round(score*100,2),"%")

# ---------------------------------------------------------
# SHAP Explainability
# ---------------------------------------------------------

class Wrapper(torch.nn.Module):

    def __init__(self,model):

        super().__init__()

        self.model=model

    def forward(self,x):

        batch=x.size(0)

        img=x[:,:3,:,:]

        dummy_ids=torch.ones(
            batch,
            64,
            dtype=torch.long,
            device=device
        )

        dummy_mask=torch.ones_like(dummy_ids)

        dummy_geo=torch.zeros(
            batch,
            2,
            device=device
        )

        out,_,_=self.model(
            img,
            dummy_ids,
            dummy_mask,
            dummy_geo
        )

        return out

wrapper=Wrapper(model)

background=torch.randn(
    5,
    3,
    224,
    224
).to(device)

explainer=shap.DeepExplainer(
    wrapper,
    background
)

sample=torch.randn(
    1,
    3,
    224,
    224
).to(device)

values=explainer.shap_values(sample)

print("SHAP Generated Successfully")

# ---------------------------------------------------------
# Save Predictions
# ---------------------------------------------------------

result=[]

with torch.no_grad():

    for img,ids,mask,geo,label in test_loader:

        img=img.to(device)
        ids=ids.to(device)
        mask=mask.to(device)
        geo=geo.to(device)

        out,_,_=model(
            img,
            ids,
            mask,
            geo
        )

        p=out.argmax(1).cpu().numpy()

        for a,b in zip(label.numpy(),p):

            result.append([
                encoder.classes_[a],
                encoder.classes_[b]
            ])

df=pd.DataFrame(
    result,
    columns=[
        "Actual",
        "Predicted"
    ]
)

df.to_csv(
    "predictions.csv",
    index=False
)

print("Prediction file saved")

# ---------------------------------------------------------
# Rule-based Chatbot
# ---------------------------------------------------------

def chatbot():

    print("="*60)
    print(" ArchaeoBrand AI Assistant ")
    print("="*60)

    while True:

        q=input("\nYou : ")

        if q.lower()=="exit":
            break

        q=q.lower()

        if "accuracy" in q:
            print("Bot :",acc)

        elif "precision" in q:
            print("Bot :",pre)

        elif "recall" in q:
            print("Bot :",rec)

        elif "f1" in q:
            print("Bot :",f1)

        elif "model" in q:
            print("Bot : DistilBERT + MobileNetV3 + SparseAE + TinyMLP")

        elif "dataset" in q:
            print("Bot :",len(df),"samples")

        elif "help" in q:
            print("""
Available Commands

accuracy
precision
recall
f1
model
dataset
exit
""")

        else:
            print("Bot : Please ask about model or evaluation.")

# ---------------------------------------------------------
# Start Chatbot
# ---------------------------------------------------------

chatbot()

print("Completed Successfully")