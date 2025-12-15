import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
# sklearn相关
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
# 可视化相关
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os

warnings.filterwarnings('ignore')

# ===================== 1. 核心配置 =====================
TRAIN_ROOT = r'C:\Users\32431\Desktop\Ma‘s\入门任务\2猫狗识别精准数据库\training_data'
TEST_ROOT = r'C:\Users\32431\Desktop\Ma‘s\入门任务\2猫狗识别精准数据库\testing_data'

CLASSES = ['cats', 'dogs']
BATCH_SIZE = 32
NUM_EPOCHS = 10
# 注意：学习率将在主函数中根据模型类型动态调整
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ===================== 2. 数据预处理与加载 =====================
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# 检查路径是否存在
if not os.path.exists(TRAIN_ROOT) or not os.path.exists(TEST_ROOT):
    print(f"❌ 错误：找不到数据集路径。\n请检查路径是否正确：\n{TRAIN_ROOT}\n{TEST_ROOT}")
    exit()

image_datasets = {
    'train': datasets.ImageFolder(root=TRAIN_ROOT, transform=data_transforms['train']),
    'test': datasets.ImageFolder(root=TEST_ROOT, transform=data_transforms['test'])
}

dataloaders = {
    'train': DataLoader(image_datasets['train'], batch_size=BATCH_SIZE, shuffle=True, num_workers=0),
    'test': DataLoader(image_datasets['test'], batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
print(f"✅ 数据加载成功 | 训练集：{dataset_sizes['train']} | 测试集：{dataset_sizes['test']}")
print(f"💻 使用设备：{DEVICE}")


# ===================== 3. 模型定义 =====================
def get_model(model_name='resnet', num_classes=2, use_pretrained=True):
    if model_name == 'resnet':
        weights = 'DEFAULT' if use_pretrained else None
        model = models.resnet18(weights=weights)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)

    elif model_name == 'swin':
        try:
            weights = 'DEFAULT' if use_pretrained else None
            model = models.swin_t(weights=weights)
            num_ftrs = model.head.in_features
            model.head = nn.Linear(num_ftrs, num_classes)
        except AttributeError:
            print("⚠️ 警告：当前torchvision版本过低，不支持Swin。自动切换回ResNet。")
            return get_model('resnet', num_classes, use_pretrained)

    elif model_name == 'svm':
        return None
    else:
        raise ValueError(f"Unknown model: {model_name}")

    return model.to(DEVICE)


# ===================== 4. PyTorch训练循环 =====================
def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=10, model_name="Model"):
    best_acc = 0.0
    best_model_wts = model.state_dict()

    print(f"\n🚀 开始训练 {model_name} ...")

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')

        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            # 简单的进度打印
            for i, (inputs, labels) in enumerate(dataloaders[phase]):
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'  {phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()

    print(f'🏆 {model_name} 最佳测试精度: {best_acc:.4f}')
    model.load_state_dict(best_model_wts)
    return model


# ===================== 5. SVM相关功能 =====================
def get_features_for_svm(dataloader, model_type='resnet'):
    print(f"⚙️ 正在提取特征 (Backbone: {model_type})...")
    if model_type == 'resnet':
        model = models.resnet18(weights='DEFAULT')
        feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])

    feature_extractor = feature_extractor.to(DEVICE)
    feature_extractor.eval()

    features = []
    labels = []

    with torch.no_grad():
        for i, (inputs, target) in enumerate(dataloader):
            if i % 10 == 0:
                print(f"\r  Processing batch {i}/{len(dataloader)}...", end="")
            inputs = inputs.to(DEVICE)
            output = feature_extractor(inputs)
            output = output.view(output.size(0), -1)
            features.append(output.cpu().numpy())
            labels.append(target.numpy())
    print("\n  特征提取完成。")
    return np.concatenate(features), np.concatenate(labels)


def train_svm(train_features, train_labels, test_features, test_labels):
    print("⚙️ 正在训练 SVM (RBF Kernel)...")
    scaler = StandardScaler()
    train_features_scaled = scaler.fit_transform(train_features)
    test_features_scaled = scaler.transform(test_features)

    svm_model = SVC(kernel='rbf', C=1.0, gamma='scale')
    svm_model.fit(train_features_scaled, train_labels)

    test_preds = svm_model.predict(test_features_scaled)
    test_acc = accuracy_score(test_labels, test_preds)
    test_prec = precision_score(test_labels, test_preds, average='macro')
    test_rec = recall_score(test_labels, test_preds, average='macro')
    test_cm = confusion_matrix(test_labels, test_preds)

    return test_acc, test_prec, test_rec, test_cm


# ===================== 6. 可视化 (无阻塞版) =====================
def plot_results(model_name, cm, acc, prec, rec, wrong_samples=None):
    # 混淆矩阵
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title(f'{model_name} (Acc={acc:.4f})')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'{model_name}_confusion_matrix.png')
    plt.close()  # 关闭图像防止显示

    # 错误样本
    if wrong_samples and len(wrong_samples) > 0:
        plt.figure(figsize=(15, 6))
        num_samples = min(5, len(wrong_samples))
        for i in range(num_samples):
            img, pred, true = wrong_samples[i]
            plt.subplot(1, num_samples, i + 1)
            img = img.permute(1, 2, 0)
            mean = torch.tensor([0.485, 0.456, 0.406])
            std = torch.tensor([0.229, 0.224, 0.225])
            img = img * std + mean
            img = torch.clamp(img, 0, 1)
            plt.imshow(img.numpy())
            plt.title(f'P:{CLASSES[pred]} / T:{CLASSES[true]}')
            plt.axis('off')
        plt.suptitle(f'{model_name} - Wrong Samples')
        plt.tight_layout()
        plt.savefig(f'{model_name}_wrong_samples.png')
        plt.close()


def plot_feature_space(train_features, train_labels, test_features, test_labels, model_name):
    # PCA可视化
    pca = PCA(n_components=2)
    all_features = np.vstack([train_features, test_features])
    features_2d = pca.fit_transform(all_features)

    # 仅画测试集，清晰一点
    test_features_2d = features_2d[len(train_labels):]

    plt.figure(figsize=(8, 6))
    for i, class_name in enumerate(CLASSES):
        idx = test_labels == i
        plt.scatter(test_features_2d[idx, 0], test_features_2d[idx, 1],
                    alpha=0.6, label=class_name, s=20)
    plt.title(f'{model_name} - Feature Space (Test Set)')
    plt.legend()
    plt.savefig(f'{model_name}_feature_space.png')
    plt.close()


# ===================== 7. 评估工具 =====================
def evaluate_model(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    wrong_identifications = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            preds = preds.cpu().numpy()
            labels = labels.cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels)

            # 收集错误
            wrong_mask = preds != labels
            if np.any(wrong_mask) and len(wrong_identifications) < 5:
                for i in np.where(wrong_mask)[0]:
                    if len(wrong_identifications) < 5:
                        wrong_identifications.append((inputs[i].cpu(), preds[i], labels[i]))

    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average='macro')
    rec = recall_score(all_labels, all_preds, average='macro')
    cm = confusion_matrix(all_labels, all_preds)

    return acc, prec, rec, cm, wrong_identifications


# ===================== 8. 主函数 =====================
def main():
    results = {}

    # --- 1. ResNet18 (基准) ---
    print("\n" + "=" * 40 + "\n🔥 阶段1: ResNet18\n" + "=" * 40)
    model_ft = get_model('resnet')
    # ResNet 使用标准学习率 1e-3
    optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.001)
    scheduler_ft = StepLR(optimizer_ft, step_size=5, gamma=0.1)

    model_ft = train_model(model_ft, dataloaders, nn.CrossEntropyLoss(), optimizer_ft, scheduler_ft, NUM_EPOCHS,
                           "ResNet18")
    acc, prec, rec, cm, wrong = evaluate_model(model_ft, dataloaders['test'], DEVICE)
    results['ResNet18'] = {'acc': acc, 'prec': prec, 'rec': rec, 'cm': cm}
    plot_results('ResNet18', cm, acc, prec, rec, wrong)

    # --- 2. SVM (ResNet特征) ---
    print("\n" + "=" * 40 + "\n🔥 阶段2: SVM (ResNet Features)\n" + "=" * 40)
    train_feat, train_lbl = get_features_for_svm(dataloaders['train'])
    test_feat, test_lbl = get_features_for_svm(dataloaders['test'])

    acc, prec, rec, cm = train_svm(train_feat, train_lbl, test_feat, test_lbl)
    results['SVM'] = {'acc': acc, 'prec': prec, 'rec': rec, 'cm': cm}
    plot_feature_space(train_feat, train_lbl, test_feat, test_lbl, 'SVM')
    plot_results('SVM', cm, acc, prec, rec)
    print(f"SVM Test Acc: {acc:.4f}")

    # --- 3. Swin-T (修复版) ---
    print("\n" + "=" * 40 + "\n🔥 阶段3: Swin-T (Optimized)\n" + "=" * 40)
    swin_model = get_model('swin')
    if swin_model:
        # [关键修改] Swin-T 对学习率敏感，使用更小的LR (5e-5) 和 AdamW
        print("ℹ️  Info: 使用 AdamW 优化器和较小的学习率 (5e-5) 以适应 Transformer")
        optimizer_swin = optim.AdamW(swin_model.parameters(), lr=0.00005, weight_decay=0.01)
        scheduler_swin = StepLR(optimizer_swin, step_size=5, gamma=0.1)

        swin_model = train_model(swin_model, dataloaders, nn.CrossEntropyLoss(), optimizer_swin, scheduler_swin,
                                 NUM_EPOCHS, "Swin-T")
        acc, prec, rec, cm, wrong = evaluate_model(swin_model, dataloaders['test'], DEVICE)
        results['Swin-T'] = {'acc': acc, 'prec': prec, 'rec': rec, 'cm': cm}
        plot_results('Swin-T', cm, acc, prec, rec, wrong)

    # --- 4. 最终总结 ---
    print("\n" + "=" * 40 + "\n📊 最终实验报告\n" + "=" * 40)
    print(f"{'Model':<15} {'Acc':<10} {'Prec':<10} {'Rec':<10}")
    print("-" * 45)
    for name, m in results.items():
        if m:
            print(f"{name:<15} {m['acc']:.4f}     {m['prec']:.4f}     {m['rec']:.4f}")

    print("\n✅ 实验结束！图片已保存，请查看文件夹。")


if __name__ == '__main__':
    main()