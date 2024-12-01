
# 貢獻指南

感謝您對 DeepLearning2 的興趣！以下是一些幫助您開始貢獻的指南：

## 開始

1. **Fork 倉庫**：將此倉庫 Fork 到您的 GitHub 帳戶。

2. **克隆 Fork**：將您的 Fork 克隆到本地機器：
   ```bash
   git clone git@github.com:your-username/DeepLearning2.git
   ```

3. **設置開發環境**：
   - 確保您已安裝 Python 3.13。
   - 創建虛擬環境：
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```
   - 安裝依賴項：
     ```bash
     pip install -r requirements.txt
     ```

## 進行更改

- **創建分支**：對於每個功能或錯誤修復，創建一個新分支：
  ```bash
  git checkout -b feature/your-feature-name
  ```

- **代碼風格**：遵循 PEP 8 的 Python 代碼風格指南。使用 `black` 進行格式化，使用 `flake8` 進行代碼檢查。

- **提交信息**：撰寫清晰、簡明的提交信息。使用祈使語氣（例如 "Add", "Fix", "Update"）。

## 提交更改

1. **推送分支**：將您的更改推送到您的 Fork：
   ```bash
   git push origin feature/your-feature-name
   ```

2. **創建拉取請求**：前往 GitHub 上的原始倉庫，從您的分支創建一個拉取請求。

3. **審查**：您的拉取請求將由維護者審查。請準備好進行更改或提供其他信息。

## 行為準則

請在貢獻此專案時遵守我們的 [行為準則](CODE_OF_CONDUCT.md)。

## 其他注意事項

- **文件**：如果您的更改影響了專案的功能或添加了新功能，請更新文件。
- **測試**：為您的代碼編寫測試。在提交拉取請求之前，確保所有測試都通過。
- **問題**：如果您發現錯誤或有功能請求，請先打開一個問題進行討論。

感謝您對 DeepLearning2 的貢獻！
