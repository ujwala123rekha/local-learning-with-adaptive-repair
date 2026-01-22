# advanced_segment_repair.py
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 128
epochs = 3
max_retries = 1
lr_segment = 1e-3
lr_predictor = 5e-3
lr_head = 5e-3

w_cos = 0.6
w_mse = 0.2
w_ce = 0.05

beta_next = 0.2

gamma_reward = 0.1

ema_alpha = 0.9


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
train_loader = DataLoader(
    datasets.MNIST('.', train=True, download=True, transform=transform),
    batch_size=batch_size, shuffle=True)
test_loader = DataLoader(
    datasets.MNIST('.', train=False, transform=transform),
    batch_size=batch_size, shuffle=False)

class Segment(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.norm = nn.LayerNorm(out_features)
        self.drop = nn.Dropout(0.2)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.norm(self.linear(x)))


class NextPredictor(nn.Module):
    """Predict the next activation (for cosine + MSE)."""
    def __init__(self, in_features, out_features):
        super().__init__()
        hid = max(in_features // 2, 32)
        self.net = nn.Sequential(
            nn.Linear(in_features, hid),
            nn.ReLU(),
            nn.Linear(hid, out_features)
        )

    def forward(self, x):
        return self.net(x)


class LocalHead(nn.Module):
    """Local classifier head for CE loss."""
    def __init__(self, in_features, n_classes=10):
        super().__init__()
        hid = max(in_features // 4, 32)
        self.net = nn.Sequential(
            nn.Linear(in_features, hid),
            nn.ReLU(),
            nn.Linear(hid, n_classes)
        )

    def forward(self, x):
        return self.net(x)


class SegmentedNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.seg1 = Segment(28*28, 256)
        self.pred1 = NextPredictor(256, 64)   # predict a2 (size 64)
        self.head1 = LocalHead(256)

        self.seg2 = Segment(256, 64)
        self.pred2 = NextPredictor(64, 64)    # predict a3 (size 64)
        self.head2 = LocalHead(64)

        self.seg3 = Segment(64, 64)
        self.classifier = nn.Linear(64, 10)

    def forward_full(self, x):
        x0 = x.view(x.size(0), -1)
        a1 = self.seg1(x0)
        a2 = self.seg2(a1)
        a3 = self.seg3(a2)
        logits = self.classifier(a3)
        return logits, [x0, a1, a2, a3]

def copy_params(module):
    return {n: p.detach().cpu().clone() for n, p in module.named_parameters()}

def restore_params(module, saved):
    with torch.no_grad():
        for n, p in module.named_parameters():
            p.copy_(saved[n].to(p.device))

model = SegmentedNet().to(device)
ce = nn.CrossEntropyLoss()

opt_seg1 = torch.optim.Adam(list(model.seg1.parameters()) + list(model.head1.parameters()) + list(model.pred1.parameters()), lr=lr_segment)
opt_seg2 = torch.optim.Adam(list(model.seg2.parameters()) + list(model.head2.parameters()) + list(model.pred2.parameters()), lr=lr_segment)
opt_seg3 = torch.optim.Adam(list(model.seg3.parameters()) + list(model.classifier.parameters()), lr=lr_segment)

prev_global_loss = None
ema_a2 = None
ema_a3 = None

print("Training advanced segment-repair model (forward-only, adaptive repair)...")

for epoch in range(epochs):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        x0 = data.view(data.size(0), -1)

        with torch.no_grad():
            a1_init = model.seg1(x0)
            a2_init = model.seg2(a1_init)
            a3_init = model.seg3(a2_init)
            logits_init = model.classifier(a3_init)
            global_loss_init = ce(logits_init, target).item()

        if ema_a2 is None:
            ema_a2 = a2_init.detach().mean(dim=0)
        else:
            ema_a2 = ema_alpha * ema_a2 + (1 - ema_alpha) * a2_init.detach().mean(dim=0)
        if ema_a3 is None:
            ema_a3 = a3_init.detach().mean(dim=0)
        else:
            ema_a3 = ema_alpha * ema_a3 + (1 - ema_alpha) * a3_init.detach().mean(dim=0)

        if prev_global_loss is None:
            reward = 1.0
        else:
            reward = 1.0 if global_loss_init < prev_global_loss else -1.0

        target_a2 = a2_init.detach()
        target_a3 = a3_init.detach()
        ema_a2_batch = ema_a2.to(device).unsqueeze(0).expand(target_a2.size(0), -1)
        ema_a3_batch = ema_a3.to(device).unsqueeze(0).expand(target_a3.size(0), -1)

        seg1_retries = 0
        improved = True
        with torch.no_grad():
            a1_cur = model.seg1(x0)
            pred_a2 = model.pred1(a1_cur)
            cos_loss = 1 - F.cosine_similarity(pred_a2, target_a2, dim=-1).mean()
            mse_loss = F.mse_loss(pred_a2, target_a2)
            logits1 = model.head1(a1_cur)
            ce_loss = ce(logits1, target)
            L1 = w_cos * cos_loss + w_mse * mse_loss + w_ce * ce_loss
            with torch.no_grad():
                a2_det = model.seg2(a1_cur.detach())
                logits2 = model.head2(a2_det)
                next_loss_proxy = ce(logits2, target)
            effective_before = L1 + beta_next * next_loss_proxy - gamma_reward * reward

        while seg1_retries < max_retries and improved:
            seg1_retries += 1
            saved_seg = copy_params(model.seg1)
            saved_pred = copy_params(model.pred1)
            saved_head = copy_params(model.head1)

            opt_seg1.zero_grad()
            a1 = model.seg1(x0)
            pred = model.pred1(a1)
            cos_loss = 1 - F.cosine_similarity(pred, target_a2, dim=-1).mean()
            mse_loss = F.mse_loss(pred, target_a2)
            logits1 = model.head1(a1)
            ce_loss = ce(logits1, target)
            L1 = w_cos * cos_loss + w_mse * mse_loss + w_ce * ce_loss


            with torch.no_grad():
                a2_from_a1 = model.seg2(a1.detach())
                logits2 = model.head2(a2_from_a1)
                next_loss_proxy = ce(logits2, target)

            effective_loss = L1 + beta_next * next_loss_proxy - gamma_reward * reward

            effective_loss.backward()
            opt_seg1.step()


            with torch.no_grad():
                a1_after = model.seg1(x0)
                pred_after = model.pred1(a1_after)
                cos_loss_after = 1 - F.cosine_similarity(pred_after, target_a2, dim=-1).mean()
                mse_loss_after = F.mse_loss(pred_after, target_a2)
                logits1_after = model.head1(a1_after)
                ce_after = ce(logits1_after, target)
                L1_after = w_cos * cos_loss_after + w_mse * mse_loss_after + w_ce * ce_after

                a2_from_a1_after = model.seg2(a1_after.detach())
                logits2_after = model.head2(a2_from_a1_after)
                next_loss_after = ce(logits2_after, target)

                effective_after = L1_after + beta_next * next_loss_after - gamma_reward * reward

            if effective_after.item() <= effective_before.item():
                effective_before = effective_after
                improved = True
            else:
                restore_params(model.seg1, saved_seg)
                restore_params(model.pred1, saved_pred)
                restore_params(model.head1, saved_head)
                improved = False


        seg2_retries = 0
        improved = True
        with torch.no_grad():
            a1_det = model.seg1(x0).detach()
            a2_cur = model.seg2(a1_det)
            pred_a3 = model.pred2(a2_cur)
            cos_loss = 1 - F.cosine_similarity(pred_a3, target_a3, dim=-1).mean()
            mse_loss = F.mse_loss(pred_a3, target_a3)
            logits2 = model.head2(a2_cur)
            ce_loss = ce(logits2, target)
            L2 = w_cos * cos_loss + w_mse * mse_loss + w_ce * ce_loss

            with torch.no_grad():
                a3_from_a2 = model.seg3(a2_cur.detach())
                logits3 = model.classifier(a3_from_a2)
                next_loss_proxy = ce(logits3, target)
            effective_before = L2 + beta_next * next_loss_proxy - gamma_reward * reward

        while seg2_retries < max_retries and improved:
            seg2_retries += 1
            saved_seg = copy_params(model.seg2)
            saved_pred = copy_params(model.pred2)
            saved_head = copy_params(model.head2)

            opt_seg2.zero_grad()
            with torch.no_grad():
                a1_det = model.seg1(x0).detach()
            a2 = model.seg2(a1_det)
            pred = model.pred2(a2)
            cos_loss = 1 - F.cosine_similarity(pred, target_a3, dim=-1).mean()
            mse_loss = F.mse_loss(pred, target_a3)
            logits2 = model.head2(a2)
            ce_loss = ce(logits2, target)
            L2 = w_cos * cos_loss + w_mse * mse_loss + w_ce * ce_loss

            with torch.no_grad():
                a3_from_a2 = model.seg3(a2.detach())
                logits3 = model.classifier(a3_from_a2)
                next_loss_proxy = ce(logits3, target)

            effective_loss = L2 + beta_next * next_loss_proxy - gamma_reward * reward
            effective_loss.backward()
            opt_seg2.step()

            with torch.no_grad():
                a1_det = model.seg1(x0).detach()
                a2_after = model.seg2(a1_det)
                pred_after = model.pred2(a2_after)
                cos_loss_after = 1 - F.cosine_similarity(pred_after, target_a3, dim=-1).mean()
                mse_loss_after = F.mse_loss(pred_after, target_a3)
                logits2_after = model.head2(a2_after)
                ce_after = ce(logits2_after, target)
                L2_after = w_cos * cos_loss_after + w_mse * mse_loss_after + w_ce * ce_after

                a3_from_a2_after = model.seg3(a2_after.detach())
                logits3_after = model.classifier(a3_from_a2_after)
                next_loss_after = ce(logits3_after, target)

                effective_after = L2_after + beta_next * next_loss_after - gamma_reward * reward

            if effective_after.item() <= effective_before.item():
                effective_before = effective_after
                improved = True
            else:
                restore_params(model.seg2, saved_seg)
                restore_params(model.pred2, saved_pred)
                restore_params(model.head2, saved_head)
                improved = False


        seg3_retries = 0
        improved = True
        with torch.no_grad():
            a2_det = model.seg2(model.seg1(x0).detach()).detach()
            a3_cur = model.seg3(a2_det)
            logits3 = model.classifier(a3_cur)
            L3 = ce(logits3, target)
            effective_before = L3 - gamma_reward * reward  # seg3 has no "next" so only reward

        while seg3_retries < max_retries and improved:
            seg3_retries += 1
            saved_seg = copy_params(model.seg3)
            saved_clf = copy_params(model.classifier)

            opt_seg3.zero_grad()
            with torch.no_grad():
                a2_det = model.seg2(model.seg1(x0).detach()).detach()
            a3 = model.seg3(a2_det)
            logits3 = model.classifier(a3)
            L3 = ce(logits3, target)
            effective_loss = L3 - gamma_reward * reward
            effective_loss.backward()
            opt_seg3.step()

            with torch.no_grad():
                a3_after = model.seg3(a2_det)
                logits3_after = model.classifier(a3_after)
                L3_after = ce(logits3_after, target)
                effective_after = L3_after - gamma_reward * reward

            if effective_after.item() <= effective_before.item():
                effective_before = effective_after
                improved = True
            else:
                restore_params(model.seg3, saved_seg)
                restore_params(model.classifier, saved_clf)
                improved = False


        with torch.no_grad():
            a1_final = model.seg1(x0)
            a2_final = model.seg2(a1_final)
            a3_final = model.seg3(a2_final)
            logits_final = model.classifier(a3_final)
            global_loss = ce(logits_final, target).item()
            preds = logits_final.argmax(dim=1)
            correct += (preds == target).sum().item()
            total += target.size(0)
            running_loss += global_loss

        prev_global_loss = global_loss

        if (batch_idx + 1) % 200 == 0:
            print(f"Epoch {epoch+1} Batch {batch_idx+1}/{len(train_loader)} | "
                  f"Loss {running_loss/(batch_idx+1):.4f} | Acc {100*correct/total:.2f}%")

    print(f"[Epoch {epoch+1}] AvgLoss {running_loss/len(train_loader):.4f} | TrainAcc {100*correct/total:.2f}%")


model.eval()
correct = 0
with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        logits, _ = model.forward_full(data)
        preds = logits.argmax(dim=1)
        correct += (preds == target).sum().item()
test_acc = 100.0 * correct / len(test_loader.dataset)
print(f"\nAdvanced repair-model Test Accuracy: {test_acc:.2f}%")

print("\nRunning standard backprop baseline...")

bp_model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(28*28, 256), nn.ReLU(),
    nn.LayerNorm(256),
    nn.Linear(256, 64), nn.ReLU(),
    nn.Linear(64, 10)
).to(device)

opt_bp = torch.optim.Adam(bp_model.parameters(), lr=1e-3)
ce = nn.CrossEntropyLoss()

for ep in range(3):
    bp_model.train()
    tot_loss, corr, tot = 0.0, 0, 0

    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        out = bp_model(data)
        loss = ce(out, target)

        opt_bp.zero_grad()
        loss.backward()
        opt_bp.step()

        tot_loss += loss.item()
        preds = out.argmax(dim=1)
        corr += (preds == target).sum().item()
        tot += target.size(0)

    print(f"BP Epoch {ep+1} | AvgLoss {tot_loss/len(train_loader):.4f} | TrainAcc {100*corr/tot:.2f}%")

bp_model.eval()
correct = 0
with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        out = bp_model(data)
        preds = out.argmax(dim=1)
        correct += (preds == target).sum().item()

print(f"Backprop Test Accuracy: {100*correct/len(test_loader.dataset):.2f}%")
