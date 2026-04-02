import os
from huggingface_hub import login, HfApi

def main():
    print("==================================================")
    print("🚀 DEPLOY LÊN HUGGINGFACE SPACES")
    print("==================================================")
    
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        print("Lấy Token tại: https://huggingface.co/settings/tokens (Nhớ chọn quyền Write)")
        token = input("👉 Dán Secret Token của bạn vào đây: ").strip()
    
    try:
        # Đăng nhập
        print("\nĐang đăng nhập...")
        login(token)
        api = HfApi()
        repo_id = "ducphuong/ObjectDetection_OCRSystem"
        
        # Upload code (bỏ qua file model nặng và thư mục git)
        print("\n📦 Đang đẩy code lên Space...")
        api.upload_folder(
            folder_path=".",
            repo_id=repo_id,
            repo_type="space",
            ignore_patterns=[
                "models/detection/*.pth",
                ".git/*",
                "__pycache__/*",
                "*.pyc",
                "outputs/*",
                ".DS_Store",
            ]
        )
        print("✅ HOÀN TẤT ĐẨY CODE!")
        
        # Upload Model Weights (riêng lẻ, ổn định hơn upload_folder)
        model_path = "models/detection/model_final.pth"
        if os.path.exists(model_path):
            size_mb = os.path.getsize(model_path) / (1024 * 1024)
            print(f"\n⏳ Đang đẩy model weights ({size_mb:.0f}MB)...")
            print("   Quá trình này có thể mất vài phút tuỳ tốc độ mạng...")
            api.upload_file(
                path_or_fileobj=model_path,
                path_in_repo=model_path,
                repo_id=repo_id,
                repo_type="space"
            )
            print("✅ HOÀN TẤT ĐẨY MODEL!")
        else:
            print(f"\n⚠️ CẢNH BÁO: Không tìm thấy file {model_path}")
            print("   Bạn cần upload thủ công qua giao diện Web của HuggingFace.")
            
        print(f"\n🔗 Link: https://huggingface.co/spaces/{repo_id}")
        print("\n⏳ Đợi HuggingFace Build Docker xong (khoảng 5-10 phút lần đầu) rồi dùng nhé!")
        
    except Exception as e:
        print(f"\n❌ LỖI: {e}")

if __name__ == "__main__":
    main()
