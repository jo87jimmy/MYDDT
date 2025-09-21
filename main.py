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
    if not os.path.exists('./outputs/'):
        os.makedirs('./outputs/')

    fin_str = "img_auc,"+run_name
    for i in image_auc:
        fin_str += "," + str(np.round(i, 3))
    fin_str += ","+str(np.round(np.mean(image_auc), 3))
    fin_str += "\n"
    fin_str += "pixel_auc,"+run_name
    for i in pixel_auc:
        fin_str += "," + str(np.round(i, 3))
    fin_str += ","+str(np.round(np.mean(pixel_auc), 3))
    fin_str += "\n"
    fin_str += "img_ap,"+run_name
    for i in image_ap:
        fin_str += "," + str(np.round(i, 3))
    fin_str += ","+str(np.round(np.mean(image_ap), 3))
    fin_str += "\n"
    fin_str += "pixel_ap,"+run_name
    for i in pixel_ap:
        fin_str += "," + str(np.round(i, 3))
    fin_str += ","+str(np.round(np.mean(pixel_ap), 3))
    fin_str += "\n"
    fin_str += "--------------------------\n"

    with open("./outputs/results.txt",'a+') as file:
        file.write(fin_str)

# =======================
# Main Pipeline
# =======================
def main():
       # 解析命令列參數
    parser = argparse.ArgumentParser()
    parser.add_argument('--category', default='bottle', type=str)  # 訓練類別
    parser.add_argument('--epochs', default=25, type=int)  # 訓練回合數
    parser.add_argument('--arch', default='wres50', type=str)  # 模型架構
    parser.add_argument('--bs', action='store', type=int, required=True)
    parser.add_argument('--lr', action='store', type=float, required=True)
    parser.add_argument('--train_bool', action='store_true', help='是否進行訓練')
    args = parser.parse_args()
    img_dim = 256  # 影像尺寸
    obj_ap_pixel_list = []
    obj_auroc_pixel_list = []
    obj_ap_image_list = []
    obj_auroc_image_list = []

    setup_seed(111)  # 固定隨機種子
    device = "cuda" if torch.cuda.is_available() else "cpu"

    path = f'./mvtec'  # 訓練資料路徑


    # 載入驗證資料集，切分方式為 "test"，同樣調整影像尺寸為 256x256
    dataset = MVTecDataset(root=path, category=args.category, split="test", resize=256)
    # 建立驗證資料的 DataLoader，設定每批次大小為 8，不打亂資料順序，使用 4 個執行緒加速載入
    dataLoader = DataLoader(dataset, batch_size=args.bs, shuffle=False, num_workers=4)

    # Load model
    # 載入模型的檢查點（checkpoint）檔案，並指定載入到的裝置（如 GPU 或 CPU）
    model_ckpt = torch.load("DRAEM_seg_large_ae_large_0.0001_800_bs8_bottle_.pckl", map_location=device,weights_only=True)
    model = ReconstructiveSubNetwork(in_channels=3, out_channels=3).to(device)
    # 建立模型的結構，輸入與輸出通道皆為 3（RGB），並移動到指定裝置上
    # model = StudentReconstructiveSubNetwork(
    #         in_channels=3,
    #         out_channels=3,
    #         base_width=64,# 壓縮後的維度
    #         teacher_base_width=128# 教師模型的維度
    #     ).to(device)
    # 將模型的參數載入至模型中，使用 checkpoint 中的 'reconstructive' 欄位
    model.load_state_dict(model_ckpt)
    # 將模型設為評估模式，停用 Dropout、BatchNorm 等訓練專用機制
    model.eval()

    # Load segmentation model
    # 載入模型的檢查點（checkpoint）檔案，並指定載入到的裝置（如 GPU 或 CPU）
    model_seg_ckpt = torch.load("DRAEM_seg_large_ae_large_0.0001_800_bs8_bottle__seg.pckl", map_location=device,weights_only=True)
    # 建立模型的結構，輸入與輸出通道皆為 3（RGB），並移動到指定裝置上
    model_seg = DiscriminativeSubNetwork(in_channels=6, out_channels=2).to(device)
    # 將模型的參數載入至模型中，使用 checkpoint 中的 'reconstructive' 欄位
    model_seg.load_state_dict(model_seg_ckpt)
    # 將模型設為評估模式，停用 Dropout、BatchNorm 等訓練專用機制
    model_seg.eval()   
    
    # 主儲存資料夾路徑
    save_root = "./save_files"

    # 若主資料夾不存在，則建立
    if not os.path.exists(save_root):
        os.makedirs(save_root)

    total_pixel_scores = np.zeros((img_dim * img_dim * len(dataset)))
    total_gt_pixel_scores = np.zeros((img_dim * img_dim * len(dataset)))
    mask_cnt = 0

    anomaly_score_gt = []
    anomaly_score_prediction = []

    display_images = torch.zeros((16 ,3 ,256 ,256)).cuda()
    display_gt_images = torch.zeros((16 ,3 ,256 ,256)).cuda()
    display_out_masks = torch.zeros((16 ,1 ,256 ,256)).cuda()
    display_in_masks = torch.zeros((16 ,1 ,256 ,256)).cuda()
    cnt_display = 0
    display_indices = np.random.randint(len(dataLoader), size=(16,))

    # 設定推論結果儲存的資料夾路徑為 save_root/inference_results
    inference_results = os.path.join(save_root, "inference_results")
    # 若資料夾不存在則建立，用來儲存推論圖像與報告
    os.makedirs(inference_results, exist_ok=True)

    for i_batch, sample_batched in enumerate(dataLoader):
        # Dataset 回傳: (image, (label, mask))
        gray_batch, (labels, true_mask) = sample_batched

        # 搬到 GPU
        gray_batch = gray_batch.cuda()
        labels = labels.cuda()
        true_mask = true_mask.cuda()

        # is_normal: 0 = good, 1 = anomaly
        is_normal = labels.detach().cpu().numpy()[0]
        anomaly_score_gt.append(is_normal)

        # 把 mask 轉成 numpy 格式 (H, W, C)
        true_mask_cv = true_mask.detach().cpu().numpy()[0, :, :, :].transpose((1, 2, 0))

        # reconstruction
        with torch.no_grad():
            gray_rec = model(gray_batch)# 只要 output，不需要 aligned_feats
            # gray_rec,_ = model(gray_batch)# 只要 output，不需要 aligned_feats
            joined_in = torch.cat((gray_rec.detach(), gray_batch), dim=1)

            # segmentation
            out_mask = model_seg(joined_in)
            out_mask_sm = torch.softmax(out_mask, dim=1)

        # 顯示指定 index 的圖片
        if i_batch in display_indices:
            #out_mask_sm[:, 0, :, :] 是正常區域的機率
            #out_mask_sm[:, 1, :, :] 是異常區域的機率
            t_mask = out_mask_sm[:, 1:, :, :]   # 取出異常區域 channel
            display_images[cnt_display] = gray_rec[0] #gray_rec 是 tensor，可以正確 .detach() 和做 indexing
            display_gt_images[cnt_display] = gray_batch[0]
            display_out_masks[cnt_display] = t_mask[0]
            display_in_masks[cnt_display] = true_mask[0]

            # 應用閾值獲得異常區塊    
            threshold = 0.5  # 可調整閾值    
            binary_mask = (out_mask_sm[:, 1, :, :] > threshold).float() 

            # 新增直接顯示圖片的程式碼  
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))  
            
            # 顯示重建圖片  
            axes[0, 0].imshow(gray_rec[0].detach().cpu().numpy().transpose(1, 2, 0))  
            axes[0, 0].set_title('Reconstructed Image')  
            axes[0, 0].axis('off')  
            
            # 顯示原始圖片  
            axes[0, 1].imshow(gray_batch[0].detach().cpu().numpy().transpose(1, 2, 0))  
            axes[0, 1].set_title('Original Image')  
            axes[0, 1].axis('off')  
            
            #檢測異常遮罩
            axes[1, 0].imshow(binary_mask[0].cpu().numpy(), cmap='Reds', alpha=0.6, vmin=0, vmax=1)
            axes[1, 0].set_title('Anomaly Detection')
            axes[1, 0].axis('off')
            
            # 顯示真實的異常遮罩  
            axes[1, 1].imshow(true_mask[0, 0].detach().cpu().numpy(), cmap='hot')  
            axes[1, 1].set_title('Ground Truth Mask')  
            axes[1, 1].axis('off')  

            # 保存圖片
            save_path = f"{inference_results}/comparison_batch{i_batch+1}.png" 
            print(f"Saving image to: {save_path}")  # 除錯用 
            plt.savefig(f"{inference_results}/comparison_batch{i_batch+1}.png") 
            plt.show()  # 加上這行來顯示圖片  
            plt.close()  
            cnt_display += 1

        # heatmap = display_out_masks.cpu().numpy()
        # fig ,axes = plt.subplots(4,4,figsize=(12,12))
        # for i in range(16):
        #     row, col = i // 4, i % 4
        #     axes[row,col].imshow(heatmap[i,0], cmap='hot',interpolation='nearest')
        #     axes[row,col].set_title(f"Heatmap {i+1}")
        #     axes[row,col].axis('off')
        # # 保存圖片
        # save_path = f"{inference_results}/comparison_batch{i_batch+1}.png" 
        # print(f"Saving image to: {save_path}")  # 除錯用 
        # plt.savefig(f"{inference_results}/comparison_batch{i_batch+1}.png") 
        # plt.show()  # 加上這行來顯示圖片  
        # plt.close()  

        out_mask_cv = out_mask_sm[0 ,1 ,: ,:].detach().cpu().numpy()

        out_mask_averaged = torch.nn.functional.avg_pool2d(out_mask_sm[: ,1: ,: ,:], 21, stride=1,
                                                            padding=21 // 2).cpu().detach().numpy()
        image_score = np.max(out_mask_averaged)

        anomaly_score_prediction.append(image_score)


        # # 顯示比較圖
        # plt.figure(figsize=(10, 5))

        # plt.subplot(1, 2, 1)
        # plt.imshow(true_mask_cv, cmap='gray')
        # plt.title("True Mask")
        # plt.axis("off")

        # plt.subplot(1, 2, 2)
        # plt.imshow(out_mask_cv, cmap='gray')
        # plt.title("Predicted Mask")
        # plt.axis("off")

        # # 存檔 + 顯示
        # save_path = f"{inference_results}/comparison_batch{i_batch+1}.png" 
        # print(f"Saving image to: {save_path}")  # 除錯用
        # plt.savefig(save_path)
        # plt.show()
        # plt.close()

        flat_true_mask = true_mask_cv.flatten()
        flat_out_mask = out_mask_cv.flatten()

        total_pixel_scores[mask_cnt * img_dim * img_dim:(mask_cnt + 1) * img_dim * img_dim] = flat_out_mask
        total_gt_pixel_scores[mask_cnt * img_dim * img_dim:(mask_cnt + 1) * img_dim * img_dim] = flat_true_mask
        mask_cnt += 1

    # # 保存定性結果  
    # for idx in range(min(cnt_display, 16)):  
    #     fig, axes = plt.subplots(1, 4, figsize=(16, 4))  
        
    #     # 原始圖像  
    #     orig_img = display_gt_images[idx].cpu().numpy().transpose(1, 2, 0)  
    #     axes[0].imshow(orig_img)  
    #     axes[0].set_title('Original Image')  
    #     axes[0].axis('off')  
        
    #     # 異常圖疊加  
    #     anomaly_overlay = orig_img.copy()  
    #     anomaly_map = display_out_masks[idx, 0].cpu().numpy()  
    #     # 創建熱力圖疊加  
    #     heatmap = plt.cm.jet(anomaly_map)[:, :, :3]  
    #     anomaly_overlay = 0.7 * orig_img + 0.3 * heatmap  
    #     axes[1].imshow(anomaly_overlay)  
    #     axes[1].set_title('Anomaly Map Overlay')  
    #     axes[1].axis('off')  
        
    #     # 異常圖  
    #     axes[2].imshow(anomaly_map, cmap='jet')  
    #     axes[2].set_title('Anomaly Map')  
    #     axes[2].axis('off')  
        
    #     # 真實標註  
    #     gt_mask = display_in_masks[idx, 0].cpu().numpy()  
    #     axes[3].imshow(gt_mask, cmap='gray')  
    #     axes[3].set_title('Ground Truth')  
    #     axes[3].axis('off')  
        
    #     plt.tight_layout()  
    #     # 存檔 + 顯示
    #     save_path = f"{inference_results}/comparison_batch{idx}.png" 
    #     print(f"Saving image to: {save_path}")  # 除錯用
    #     plt.savefig(save_path)
    #     plt.show()
    #     plt.close()

    anomaly_score_prediction = np.array(anomaly_score_prediction)
    anomaly_score_gt = np.array(anomaly_score_gt)
    auroc = roc_auc_score(anomaly_score_gt, anomaly_score_prediction)
    ap = average_precision_score(anomaly_score_gt, anomaly_score_prediction)

    total_gt_pixel_scores = total_gt_pixel_scores.astype(np.uint8)
    total_gt_pixel_scores = total_gt_pixel_scores[:img_dim * img_dim * mask_cnt]
    total_pixel_scores = total_pixel_scores[:img_dim * img_dim * mask_cnt]
    auroc_pixel = roc_auc_score(total_gt_pixel_scores, total_pixel_scores)
    ap_pixel = average_precision_score(total_gt_pixel_scores, total_pixel_scores)
    obj_ap_pixel_list.append(ap_pixel)
    obj_auroc_pixel_list.append(auroc_pixel)
    obj_auroc_image_list.append(auroc)
    obj_ap_image_list.append(ap)
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

    write_results_to_file(args.category, obj_auroc_image_list, obj_auroc_pixel_list, obj_ap_image_list, obj_ap_pixel_list)


# =======================
# Run pipeline
# =======================
if __name__ == "__main__":
    main()
