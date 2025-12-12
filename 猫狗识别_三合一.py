import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
# sklearn相关（SVM训练+指标计算）
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
# 可视化相关
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.decomposition import PCA  # 用于特征降维可视化

warnings.filterwarnings('ignore')  # 屏蔽无关警告

# ===================== 1. 核心配置（适配你的数据集路径） =====================
# 训练/测试数据集根路径（Windows绝对路径，用原始字符串避免转义）
TRAIN_ROOT = r'C:\Users\32431\Desktop\Ma‘s\入门任务\2猫狗识别精准数据库\training_data'
TEST_ROOT = r'C:\Users\32431\Desktop\Ma‘s\入门任务\2猫狗识别精准数据库\testing_data'
# 类别名称（与数据集文件夹名对应）
CLASSES = ['cats', 'dogs']
# 训练超参数
BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 自动适配CPU/GPU

# ===================== 2. 数据预处理与加载（Windows兼容） =====================
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

# 加载数据集
image_datasets = {
    'train': datasets.ImageFolder(root=TRAIN_ROOT, transform=data_transforms['train']),
    'test': datasets.ImageFolder(root=TEST_ROOT, transform=data_transforms['test'])
}

dataloaders = {
    'train': DataLoader(image_datasets['train'], batch_size=BATCH_SIZE, shuffle=True, num_workers=0),
    'test': DataLoader(image_datasets['test'], batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
print(f"训练集样本数：{dataset_sizes['train']} | 测试集样本数：{dataset_sizes['test']}")
print(f"使用设备：{DEVICE}")


# ===================== 3. 模型定义（ResNet18/Swin-T/SVM） =====================
def get_model(model_name='resnet', num_classes=2, use_pretrained=True):
    """获取指定类型的模型"""

    if model_name == 'resnet':
        # ResNet18模型
        weights = 'DEFAULT' if use_pretrained else None
        model = models.resnet18(weights=weights)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)

    elif model_name == 'swin':
        # Swin-T模型
        try:
            weights = 'DEFAULT' if use_pretrained else None
            model = models.swin_t(weights=weights)
            num_ftrs = model.head.in_features
            model.head = nn.Linear(num_ftrs, num_classes)
        except AttributeError:
            print("警告：当前torchvision版本不支持swin_t模型，将使用ResNet18替代")
            return get_model('resnet', num_classes, use_pretrained)

    elif model_name == 'svm':
        # SVM模型（返回None，因为SVM在sklearn中单独训练）
        return None

    else:
        raise ValueError(f"不支持的模型：{model_name}，可选'resnet'/'swin'/'svm'")

    return model.to(DEVICE)


# ===================== 4. PyTorch模型训练循环 =====================
def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=10):
    """训练PyTorch模型"""
    best_acc = 0.0
    best_model_wts = model.state_dict()

    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch + 1}/{num_epochs}')
        print('-' * 50)

        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
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

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()

    model.load_state_dict(best_model_wts)
    print(f'\n最佳测试精度: {best_acc:.4f}')
    return model


# ===================== 5. SVM相关功能 =====================
def get_features_for_svm(dataloader, model_type='resnet', use_pretrained=True):
    """提取图像特征用于SVM训练"""

    # 加载特征提取模型
    if model_type == 'resnet':
        model = models.resnet18(weights='DEFAULT' if use_pretrained else None)
        # 移除最后一层全连接层
        feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])
    elif model_type == 'swin':
        model = models.swin_t(weights='DEFAULT' if use_pretrained else None)
        # 移除分类头
        feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])
    else:
        raise ValueError("仅支持resnet或swin作为特征提取器")

    feature_extractor = feature_extractor.to(DEVICE)
    feature_extractor.eval()

    features = []
    labels = []

    with torch.no_grad():
        for inputs, target in dataloader:
            inputs = inputs.to(DEVICE)
            output = feature_extractor(inputs)
            output = output.view(output.size(0), -1)  # 展平特征
            features.append(output.cpu().numpy())
            labels.append(target.numpy())

    return np.concatenate(features), np.concatenate(labels)


def train_svm(train_features, train_labels, test_features, test_labels, kernel='rbf'):
    """训练并评估SVM模型"""

    # 特征标准化
    scaler = StandardScaler()
    train_features_scaled = scaler.fit_transform(train_features)
    test_features_scaled = scaler.transform(test_features)

    # 训练SVM
    svm_model = SVC(kernel=kernel, C=1.0, gamma='scale', probability=True)
    svm_model.fit(train_features_scaled, train_labels)

    # 预测
    train_preds = svm_model.predict(train_features_scaled)
    test_preds = svm_model.predict(test_features_scaled)

    # 计算指标
    train_acc = accuracy_score(train_labels, train_preds)
    test_acc = accuracy_score(test_labels, test_preds)
    test_prec = precision_score(test_labels, test_preds, average='macro')
    test_rec = recall_score(test_labels, test_preds, average='macro')
    test_cm = confusion_matrix(test_labels, test_preds)

    print(f"SVM训练集准确率: {train_acc:.4f}")
    print(f"SVM测试集准确率: {test_acc:.4f}")
    print(f"SVM测试集精确率: {test_prec:.4f}")
    print(f"SVM测试集召回率: {test_rec:.4f}")

    return svm_model, test_acc, test_prec, test_rec, test_cm, test_preds


# ===================== 6. 结果分析与可视化 =====================
def plot_results(model_name, cm, acc, prec, rec, wrong_samples=None):
    """绘制结果可视化"""

    # 1. 绘制混淆矩阵
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title(f'{model_name} - 混淆矩阵 (Acc={acc:.4f}, Prec={prec:.4f}, Rec={rec:.4f})')
    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    plt.tight_layout()
    plt.savefig(f'{model_name}_confusion_matrix.png')
    plt.show()

    # 2. 可视化错误识别样本
    if wrong_samples and len(wrong_samples) > 0:
        plt.figure(figsize=(15, 6))
        num_samples = min(5, len(wrong_samples))
        for i in range(num_samples):
            img, pred, true = wrong_samples[i]
            plt.subplot(1, num_samples, i + 1)
            # 反归一化
            img = img.permute(1, 2, 0)
            mean = torch.tensor([0.485, 0.456, 0.406])
            std = torch.tensor([0.229, 0.224, 0.225])
            img = img * std + mean
            img = torch.clamp(img, 0, 1)
            plt.imshow(img.numpy())
            plt.title(f'Pred: {CLASSES[pred]}\nTrue: {CLASSES[true]}')
            plt.axis('off')
        plt.suptitle(f'{model_name} - 错误识别样本')
        plt.tight_layout()
        plt.savefig(f'{model_name}_wrong_samples.png')
        plt.show()


def plot_feature_space(train_features, train_labels, test_features, test_labels, model_name):
    """可视化特征空间"""
    # 使用PCA降维到2D
    pca = PCA(n_components=2)

    # 合并训练和测试特征
    all_features = np.vstack([train_features, test_features])
    all_labels = np.concatenate([train_labels, test_labels])

    # 应用PCA
    features_2d = pca.fit_transform(all_features)

    # 分开训练和测试
    train_features_2d = features_2d[:len(train_labels)]
    test_features_2d = features_2d[len(train_labels):]

    plt.figure(figsize=(12, 5))

    # 训练集特征
    plt.subplot(1, 2, 1)
    for i, class_name in enumerate(CLASSES):
        idx = train_labels == i
        plt.scatter(train_features_2d[idx, 0], train_features_2d[idx, 1],
                    alpha=0.6, label=class_name, s=20)
    plt.title(f'{model_name} - 训练集特征空间(PCA)')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.legend()

    # 测试集特征
    plt.subplot(1, 2, 2)
    for i, class_name in enumerate(CLASSES):
        idx = test_labels == i
        plt.scatter(test_features_2d[idx, 0], test_features_2d[idx, 1],
                    alpha=0.6, label=class_name, s=20)
    plt.title(f'{model_name} - 测试集特征空间(PCA)')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'{model_name}_feature_space.png')
    plt.show()


# ===================== 7. 模型评估 =====================
def evaluate_model(model, dataloader, device, classes, model_type='pytorch'):
    """评估模型性能"""

    if model_type == 'svm':
        # SVM评估已经在train_svm函数中完成
        return None, None, None, None, []

    model.eval()
    all_preds = []
    all_labels = []
    wrong_identifications = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)

            if model_type == 'pytorch':
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
            else:
                raise ValueError(f"不支持的模型类型: {model_type}")

            preds_cpu = preds.cpu().numpy()
            labels_cpu = labels.cpu().numpy()

            all_preds.extend(preds_cpu)
            all_labels.extend(labels_cpu)

            # 收集错误样本
            if len(wrong_identifications) < 5:
                wrong_mask = preds_cpu != labels_cpu
                for i in range(min(5, sum(wrong_mask))):
                    idx = np.where(wrong_mask)[0][i]
                    wrong_identifications.append((
                        inputs[idx].cpu(),
                        preds_cpu[idx],
                        labels_cpu[idx]
                    ))

    # 计算评估指标
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average='macro')
    rec = recall_score(all_labels, all_preds, average='macro')
    cm = confusion_matrix(all_labels, all_preds)

    return acc, prec, rec, cm, wrong_identifications


# ===================== 8. 主函数 - 运行三个模型 =====================
def main():
    """主函数：依次运行三个模型"""

    results = {}  # 存储所有模型的结果

    # -------------------- 1. 训练和评估ResNet18 --------------------
    print("\n" + "=" * 60)
    print("阶段1：训练和评估ResNet18模型")
    print("=" * 60)

    resnet_model = get_model('resnet', num_classes=2, use_pretrained=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(resnet_model.parameters(), lr=LEARNING_RATE)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

    # 训练模型
    resnet_model = train_model(resnet_model, dataloaders, criterion, optimizer,
                               scheduler, num_epochs=NUM_EPOCHS)

    # 保存模型
    torch.save(resnet_model.state_dict(), 'best_resnet18.pth')

    # 评估模型
    acc, prec, rec, cm, wrong_samples = evaluate_model(
        resnet_model, dataloaders['test'], DEVICE, CLASSES, 'pytorch'
    )

    results['ResNet18'] = {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'confusion_matrix': cm
    }

    print(f"ResNet18 测试集指标：")
    print(f"准确率: {acc:.4f} | 精确率: {prec:.4f} | 召回率: {rec:.4f}")

    # 可视化结果
    plot_results('ResNet18', cm, acc, prec, rec, wrong_samples)

    # -------------------- 2. 训练和评估SVM --------------------
    print("\n" + "=" * 60)
    print("阶段2：使用ResNet18特征训练SVM")
    print("=" * 60)

    # 提取特征
    print("提取训练集特征...")
    train_features, train_labels = get_features_for_svm(dataloaders['train'], 'resnet', True)
    print("提取测试集特征...")
    test_features, test_labels = get_features_for_svm(dataloaders['test'], 'resnet', True)

    # 训练SVM
    svm_model, svm_acc, svm_prec, svm_rec, svm_cm, svm_preds = train_svm(
        train_features, train_labels, test_features, test_labels, 'rbf'
    )

    results['SVM'] = {
        'accuracy': svm_acc,
        'precision': svm_prec,
        'recall': svm_rec,
        'confusion_matrix': svm_cm
    }

    # 可视化SVM特征空间
    plot_feature_space(train_features, train_labels, test_features, test_labels, 'SVM')
    plot_results('SVM', svm_cm, svm_acc, svm_prec, svm_rec)

    # -------------------- 3. 训练和评估Swin-T --------------------
    print("\n" + "=" * 60)
    print("阶段3：训练和评估Swin-T模型")
    print("=" * 60)

    swin_model = get_model('swin', num_classes=2, use_pretrained=True)

    if swin_model is not None:
        swin_criterion = nn.CrossEntropyLoss()
        swin_optimizer = optim.Adam(swin_model.parameters(), lr=LEARNING_RATE)
        swin_scheduler = StepLR(swin_optimizer, step_size=5, gamma=0.1)

        # 训练Swin-T
        swin_model = train_model(swin_model, dataloaders, swin_criterion,
                                 swin_optimizer, swin_scheduler, num_epochs=NUM_EPOCHS)

        # 保存模型
        torch.save(swin_model.state_dict(), 'best_swin_t.pth')

        # 评估模型
        swin_acc, swin_prec, swin_rec, swin_cm, swin_wrong = evaluate_model(
            swin_model, dataloaders['test'], DEVICE, CLASSES, 'pytorch'
        )

        results['Swin-T'] = {
            'accuracy': swin_acc,
            'precision': swin_prec,
            'recall': swin_rec,
            'confusion_matrix': swin_cm
        }

        print(f"Swin-T 测试集指标：")
        print(f"准确率: {swin_acc:.4f} | 精确率: {swin_prec:.4f} | 召回率: {swin_rec:.4f}")

        # 可视化结果
        plot_results('Swin-T', swin_cm, swin_acc, swin_prec, swin_rec, swin_wrong)
    else:
        print("Swin-T模型加载失败，跳过此模型")
        results['Swin-T'] = None

    # -------------------- 4. 结果对比 --------------------
    print("\n" + "=" * 60)
    print("阶段4：三种模型性能对比")
    print("=" * 60)

    print(f"\n{'模型':<15} {'准确率':<12} {'精确率':<12} {'召回率':<12}")
    print("-" * 60)

    for model_name, metrics in results.items():
        if metrics is not None:
            print(f"{model_name:<15} {metrics['accuracy']:.4f}      "
                  f"{metrics['precision']:.4f}      {metrics['recall']:.4f}")

    # 找出最佳模型
    if results['Swin-T'] is not None:
        valid_results = results
    else:
        valid_results = {k: v for k, v in results.items() if v is not None}

    best_model = max(valid_results, key=lambda x: valid_results[x]['accuracy'])
    print("\n" + "=" * 60)
    print(f"🏆 最佳模型: {best_model}, 准确率: {valid_results[best_model]['accuracy']:.4f}")
    print("=" * 60)

    # 保存所有结果到文件
    import json
    with open('model_results.json', 'w') as f:
        # 将numpy数组转换为列表以便JSON序列化
        serializable_results = {}
        for model_name, metrics in results.items():
            if metrics is not None:
                serializable_results[model_name] = {
                    'accuracy': float(metrics['accuracy']),
                    'precision': float(metrics['precision']),
                    'recall': float(metrics['recall']),
                    'confusion_matrix': metrics['confusion_matrix'].tolist()
                }
            else:
                serializable_results[model_name] = None
        json.dump(serializable_results, f, indent=4)

    print("\n✅ 所有模型训练完成！")
    print("📊 结果已保存到: model_results.json")
    print("📸 可视化图表已保存为PNG文件")


if __name__ == '__main__':
    main()