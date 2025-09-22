import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T
from PIL import Image
import numpy as np
from sklearn.metrics import roc_auc_score
import random  # 亂數控制
import argparse  # 命令列參數處理
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_auc_score, average_precision_score
from model_unet import ReconstructiveSubNetwork,DiscriminativeSubNetwork,StudentReconstructiveSubNetwork
import matplotlib.pyplot as plt
import cv2
from torchvision.utils import make_grid, save_image
import torch.nn.functional as F

def setup_seed(seed):
    # 設定隨機種子，確保實驗可重現
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True  # 保證結果可重現
    torch.backends.cudnn.benchmark = False  # 關閉自動最佳化搜尋

# =======================
# Dataset
# =======================
# 定義 MVTecDataset 類別，繼承自 PyTorch 的 Dataset，用於載入 MVTec 異常偵測資料集
class MVTecDataset(torch.utils.data.Dataset):
    def __init__(self, root, category="bottle", split="train", resize=256):
        # 設定資料根目錄
        self.root = root
        # 指定類別（如 bottle、capsule 等）
        self.category = category
        # 指定資料切分方式（train 或 test）
        self.split = split
        # 設定影像資料夾路徑
        self.img_dir = os.path.join(root, category, split)
        # 設定 ground truth 遮罩資料夾路徑
        self.gt_dir = os.path.join(root, category, "ground_truth")
        # 初始化影像路徑、標籤與遮罩路徑的清單
        self.data, self.labels, self.masks = [], [], []

        # 遍歷影像資料夾中的所有子類別（如 good、broken_small 等）
        for defect_type in sorted(os.listdir(self.img_dir)):
            img_folder = os.path.join(self.img_dir, defect_type)
            # 若不是資料夾則跳過（排除非類別資料）
            if not os.path.isdir(img_folder):
                continue
            # 遍歷該類別資料夾中的所有影像檔案
            for f in sorted(os.listdir(img_folder)):
                # 若為正常類別（good），則不含遮罩
                img_path = os.path.join(img_folder, f)
                if defect_type == "good":
                    self.data.append(img_path)# 儲存影像路徑
                    self.labels.append(0) # 標記為正常樣本（label=0）
                    self.masks.append(None) # 遮罩為 None
                else:
                    # 若為異常類別，則根據影像檔名推導遮罩路徑
                    mask_path = os.path.join(self.gt_dir, defect_type, f.replace(".png","_mask.png"))
                    self.data.append(img_path) # 儲存影像路徑
                    self.labels.append(1) # 標記為異常樣本（label=1）
                    self.masks.append(mask_path) # 儲存遮罩路徑

        # 定義影像的轉換流程：調整大小並轉為 Tensor
        self.transform = T.Compose([
            T.Resize((resize, resize)),# 調整影像尺寸為指定大小
            T.ToTensor()# 轉換為 PyTorch Tensor 格式
        ])

        # 定義遮罩的轉換流程：使用最近鄰插值法調整大小並轉為 Tensor
        self.mask_transform = T.Compose([
            T.Resize((resize, resize), interpolation=Image.NEAREST),# 避免遮罩模糊
            T.ToTensor()# 轉換為 Tensor 格式
        ])

    # 回傳資料集的總長度（樣本數）
    def __len__(self):
        return len(self.data)
    # 定義資料集的索引存取方式
    def __getitem__(self, idx):
        img = Image.open(self.data[idx]).convert("RGB")# 讀取指定索引的影像並轉為 RGB 格式
        img = self.transform(img)# 套用影像轉換流程（resize + tensor）
        label = self.labels[idx] # 取得該影像的標籤（0 或 1）
        mask_path = self.masks[idx]# 取得該影像對應的遮罩路徑（可能為 None）

        # 若遮罩為 None（正常樣本），則建立全黑遮罩
        if mask_path is None:
            mask = torch.zeros((1, img.shape[1], img.shape[2]))# 單通道全 0 遮罩
        else:
            mask = Image.open(mask_path).convert("L")# 否則載入遮罩圖並轉為灰階
            mask = self.mask_transform(mask)# 套用遮罩轉換流程（resize + tensor）
            mask = (mask>0.5).float()# 將遮罩二值化（大於 0.5 的視為異常區域）
         # 回傳影像、標籤與遮罩（標籤為 tensor，遮罩為 float tensor）
        return img, (torch.tensor(label, dtype=torch.long), mask)

def write_results_to_file(run_name, image_auc, pixel_auc, image_ap, pixel_ap):
    # 檢查是否存在輸出資料夾，若不存在則建立
    if not os.path.exists('./outputs/'):
        os.makedirs('./outputs/')

    # 構建結果字串，依序記錄不同指標
    fin_str = "img_auc,"+run_name
    for i in image_auc:  # 逐一加入影像層級的 AUC
        fin_str += "," + str(np.round(i, 3))
    fin_str += ","+str(np.round(np.mean(image_auc), 3))  # 加入平均值
    fin_str += "\n"
    fin_str += "pixel_auc,"+run_name
    for i in pixel_auc:  # 逐一加入像素層級的 AUC
        fin_str += "," + str(np.round(i, 3))
    fin_str += ","+str(np.round(np.mean(pixel_auc), 3))
    fin_str += "\n"
    fin_str += "img_ap,"+run_name
    for i in image_ap:  # 逐一加入影像層級的 AP
        fin_str += "," + str(np.round(i, 3))
    fin_str += ","+str(np.round(np.mean(image_ap), 3))
    fin_str += "\n"
    fin_str += "pixel_ap,"+run_name
    for i in pixel_ap:  # 逐一加入像素層級的 AP
        fin_str += "," + str(np.round(i, 3))
    fin_str += ","+str(np.round(np.mean(pixel_ap), 3))
    fin_str += "\n"
    fin_str += "--------------------------\n"

    # 將結果寫入檔案（附加模式）
    with open("./outputs/results.txt",'a+') as file:
        file.write(fin_str)

# =======================
# Main Pipeline 主流程
# =======================
def main():
    # 解析命令列參數
    parser = argparse.ArgumentParser()
    parser.add_argument('--category', default='bottle', type=str)  # 訓練類別
    parser.add_argument('--epochs', default=25, type=int)  # 訓練回合數
    parser.add_argument('--arch', default='wres50', type=str)  # 模型架構
    parser.add_argument('--bs', action='store', type=int, required=True)  # batch size
    parser.add_argument('--lr', action='store', type=float, required=True)  # 學習率
    parser.add_argument('--train_bool', action='store_true', help='是否進行訓練')  # 是否訓練
    args = parser.parse_args()

    img_dim = 256  # 設定影像尺寸
    obj_ap_pixel_list = []       # 紀錄像素層級 AP
    obj_auroc_pixel_list = []    # 紀錄像素層級 AUC
    obj_ap_image_list = []       # 紀錄影像層級 AP
    obj_auroc_image_list = []    # 紀錄影像層級 AUC

    setup_seed(111)  # 固定隨機種子，確保實驗可重現
    device = "cuda" if torch.cuda.is_available() else "cpu"  # 選擇 GPU 或 CPU

    path = f'./mvtec'  # 訓練資料路徑

    # 載入驗證資料集 (測試集)，影像大小為 256x256
    dataset = MVTecDataset(root=path, category=args.category, split="test", resize=256)
    # 建立 DataLoader，每批次大小為 args.bs，不打亂，使用 4 個執行緒載入
    dataLoader = DataLoader(dataset, batch_size=args.bs, shuffle=False, num_workers=4)

    # 載入重建模型 checkpoint
    model_ckpt = torch.load("DRAEM_seg_large_ae_large_0.0001_800_bs8_capsule_.pckl", map_location=device, weights_only=True)
    model = ReconstructiveSubNetwork(in_channels=3, out_channels=3).to(device)  # 初始化模型結構
    model.load_state_dict(model_ckpt)  # 載入權重
    model.eval()  # 設為推論模式

    # 載入分割模型 checkpoint
    model_seg_ckpt = torch.load("DRAEM_seg_large_ae_large_0.0001_800_bs8_capsule__seg.pckl", map_location=device, weights_only=True)
    model_seg = DiscriminativeSubNetwork(in_channels=6, out_channels=2).to(device)  # 初始化分割模型
    model_seg.load_state_dict(model_seg_ckpt)  # 載入權重
    model_seg.eval()

    # 建立主存檔資料夾
    save_root = "./save_files"
    if not os.path.exists(save_root):
        os.makedirs(save_root)

    # 建立全域統計用的陣列
    total_pixel_scores = np.zeros((img_dim * img_dim * len(dataset)))     # 模型預測分數
    total_gt_pixel_scores = np.zeros((img_dim * img_dim * len(dataset)))  # 真實標註
    mask_cnt = 0  # 計算 mask 數量

    anomaly_score_gt = []          # 儲存 ground truth (0=正常,1=異常)
    anomaly_score_prediction = []  # 儲存預測分數

    # 儲存顯示用影像
    display_images = torch.zeros((16 ,3 ,256 ,256)).cuda()
    display_gt_images = torch.zeros((16 ,3 ,256 ,256)).cuda()
    display_out_masks = torch.zeros((16 ,1 ,256 ,256)).cuda()
    display_in_masks = torch.zeros((16 ,1 ,256 ,256)).cuda()
    cnt_display = 0
    # 隨機選 16 個批次索引
    display_indices = np.random.randint(len(dataLoader), size=(16,))

    # 推論結果輸出資料夾
    inference_results = os.path.join(save_root, "inference_results")
    os.makedirs(inference_results, exist_ok=True)

    # =========================
    # 推論迴圈
    # =========================
    for i_batch, sample_batched in enumerate(dataLoader):
        gray_batch, (labels, true_mask) = sample_batched  # 取得影像、標籤與 mask

        # 搬到 GPU
        gray_batch = gray_batch.cuda()
        labels = labels.cuda()
        true_mask = true_mask.cuda()

        # Convert tensor to a numpy array and move it to the CPU
        image = gray_batch.permute(0, 2, 3, 1).cpu().numpy()

        # Display all images in the batch
        for i in range(image.shape[0]):
            plt.imshow(image[i], cmap='gray')
            plt.title('Original Image')
            save_path = f"{inference_results}/Original Image{i}.png"
            print(f"Saving Original Image to: {save_path}")  # 除錯訊息
            plt.savefig(save_path)
            plt.show()

        # ground truth: 0=正常,1=異常
        is_normal = labels.detach().cpu().numpy()[0]
        anomaly_score_gt.append(is_normal)

        # 轉成 numpy 格式 (H, W, C)
        true_mask_cv = true_mask.detach().cpu().numpy()[0, :, :, :].transpose((1, 2, 0))

        # 模型推論
        with torch.no_grad():
            gray_rec = model(gray_batch)  #gray_rec:重建影像，gray_batch:原始圖像
            # gray_rec[重建影像](B, 3, H, W)，joined_in[拼接原始輸入與重建影像](B, 6, H, W)
            joined_in = torch.cat((gray_rec.detach(), gray_batch), dim=1)  # 拼接輸入
            #將重建網路的3通道輸出和原始影像輸入給分割網路
            out_mask = model_seg(joined_in)  # 分割模型推論，逐像素分割，差異越大的地方，越可能是「異常區域」
            # Softmax 機率圖
            #把 logits 轉成機率。shape 還是 (B, C, H, W)，但現在每個 pixel 的 C 個通道加總 = 1。
            #例如某個 pixel：
            #out_mask_sm[b, :, y, x] = [0.95, 0.05] → 95% 正常, 5% 異常
            out_mask_sm = torch.softmax(out_mask, dim=1)  # out_mask_sm shape 是 (B, C, H, W)，B = batch size，C = channel 數(分類數)，H, W = 高寬

        # 若該批次在隨機顯示列表中，則繪製結果
        if i_batch in display_indices:
            t_mask = out_mask_sm[:, 1:, :, :]   # 取異常機率圖
            display_images[cnt_display] = gray_rec[0]
            display_gt_images[cnt_display] = gray_batch[0]
            display_out_masks[cnt_display] = t_mask[0]
            display_in_masks[cnt_display] = true_mask[0]
            cnt_display += 1

        # 計算 pixel-level score
        out_mask_cv = out_mask_sm[0 ,1 ,: ,:].detach().cpu().numpy()# 第0張圖的單通道
        # 直接顯示概率遮罩
        plt.imshow(out_mask_cv)
        save_path = f"{inference_results}/out_mask_cv{i_batch+1}.png"
        print(f"Saving out_mask_cv image to: {save_path}")  # 除錯訊息
        plt.savefig(save_path)
        plt.show()

        # 或者轉換為 0-255 範圍並保存
        mask_image = (out_mask_cv * 255).astype(np.uint8)
        mask_save_path = f"{inference_results}/defect_mask_batch{i_batch+1}.png"
        cv2.imwrite(mask_save_path, mask_image)
        print(f"Saving mask to: {mask_save_path}")

        out_mask_averaged = torch.nn.functional.avg_pool2d(out_mask_sm[: ,1: ,: ,:], 21, stride=1,
                                                           padding=21 // 2).cpu().detach().numpy()
        image_score = np.max(out_mask_averaged)  # 單張影像分數
        anomaly_score_prediction.append(image_score)

        # 攤平成 1D，儲存像素分數
        flat_true_mask = true_mask_cv.flatten()
        flat_out_mask = out_mask_cv.flatten()
        total_pixel_scores[mask_cnt * img_dim * img_dim:(mask_cnt + 1) * img_dim * img_dim] = flat_out_mask
        total_gt_pixel_scores[mask_cnt * img_dim * img_dim:(mask_cnt + 1) * img_dim * img_dim] = flat_true_mask
        mask_cnt += 1

    # =========================
    # 評估指標計算
    # =========================
    anomaly_score_prediction = np.array(anomaly_score_prediction)
    anomaly_score_gt = np.array(anomaly_score_gt)
    auroc = roc_auc_score(anomaly_score_gt, anomaly_score_prediction)  # 影像層級 AUC
    ap = average_precision_score(anomaly_score_gt, anomaly_score_prediction)  # 影像層級 AP

    total_gt_pixel_scores = total_gt_pixel_scores.astype(np.uint8)
    total_gt_pixel_scores = total_gt_pixel_scores[:img_dim * img_dim * mask_cnt]
    total_pixel_scores = total_pixel_scores[:img_dim * img_dim * mask_cnt]
    auroc_pixel = roc_auc_score(total_gt_pixel_scores, total_pixel_scores)  # 像素層級 AUC
    ap_pixel = average_precision_score(total_gt_pixel_scores, total_pixel_scores)  # 像素層級 AP

    # 儲存到列表
    obj_ap_pixel_list.append(ap_pixel)
    obj_auroc_pixel_list.append(auroc_pixel)
    obj_auroc_image_list.append(auroc)
    obj_ap_image_list.append(ap)

    # 輸出結果
    print(args.category)
    print("AUC Image:  " +str(auroc))
    print("AP Image:  " +str(ap))
    print("AUC Pixel:  " +str(auroc_pixel))
    print("AP Pixel:  " +str(ap_pixel))
    print("==============================")
    print(args.category)
    print("AUC Image mean:  " + str(np.mean(obj_auroc_image_list)))
    print("AP Image mean:  " + str(np.mean(obj_ap_image_list)))
    print("AUC Pixel mean:  " + str(np.mean(obj_auroc_pixel_list)))
    print("AP Pixel mean:  " + str(np.mean(obj_ap_pixel_list)))

    # 將結果寫入檔案
    write_results_to_file(args.category, obj_auroc_image_list, obj_auroc_pixel_list, obj_ap_image_list, obj_ap_pixel_list)


# =======================
# Run pipeline
# =======================
if __name__ == "__main__":
    main()
