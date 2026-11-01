# 给 GitHub 仓库设置 About（简介 / 主题 / 网址）
#
# 用法：
#   1) 安装 GitHub CLI：winget install GitHub.cli
#   2) 登录：gh auth login           （选 GitHub.com → HTTPS → 用浏览器登录）
#   3) 运行本脚本：pwsh -File scripts/set_repo_about.ps1
#
# 也可以不开脚本，直接在仓库页右侧 About 的齿轮里粘贴 DESCRIPTION 那段。

$ErrorActionPreference = "Stop"

$repo = "SinWrong/picture_processing_tool"

$description = "图片水印分析与去除工具（纯 Python + numpy + Pillow，无需 OpenCV）：自动检测 / 鼠标框选 / 水印样本匹配，支持批量处理与处理前后左右对比。Watermark removal toolkit — auto-detect, box-select, sample matching, batch, before/after preview."

$topics = @(
  "watermark-removal",
  "image-processing",
  "inpainting",
  "python",
  "numpy",
  "pillow",
  "tkinter",
  "gui",
  "cli",
  "image-editing"
)

gh --version *> $null
if ($LASTEXITCODE -ne 0) {
  Write-Host "没有找到 gh（GitHub CLI）。先运行： winget install GitHub.cli" -ForegroundColor Yellow
  Write-Host "或者手动设置，描述文本见本文件里的 `$description`。" -ForegroundColor Yellow
  exit 1
}

gh repo edit $repo --description $description --homepage "https://github.com/$repo"
foreach ($t in $topics) {
  gh repo edit $repo --add-topic $t
}

Write-Host "完成，打开 https://github.com/$repo 看看右侧 About。" -ForegroundColor Green
