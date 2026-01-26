#!/bin/bash
#
# 这个脚本用于在 Docker 容器中快速设置 Conda 环境。
#
# #######################################################
# #               !! 非常重要 !!                       #
# # 你必须 "source" 这个脚本，而不是直接运行它。        #
# #                                                   #
# # 正确用法:  source setup_conda.sh                  #
# # 或者:      . setup_conda.sh                        #
# #                                                   #
# # 错误用法:  ./setup_conda.sh                       #
# #######################################################
#

# 1. 这是你指定的 Conda 路径
CONDA_PATH="/workspace/anaconda3"

# 2. 运行 'conda init'
#    这会修改你的 ~/.bashrc 文件，以便在下次登录时自动设置 Conda。
#    我们检查一下，防止重复添加。
if ! grep -q "# >>> conda initialize >>>" ~/.bashrc; then
    echo "正在将 Conda 初始化代码添加到 ~/.bashrc (用于未来的会话)..."
    "$CONDA_PATH/bin/conda" init bash
else
    echo "Conda 已经在 ~/.bashrc 中初始化。"
fi

# 3. 为当前会话激活 Conda
#    'conda init' 只修改了 .bashrc，不会影响当前 shell。
#    我们必须手动 source conda.sh 来定义 'conda' shell 函数。
echo "正在为当前会话加载 Conda..."

if [ -f "$CONDA_PATH/etc/profile.d/conda.sh" ]; then
    # Sourcing 这个脚本会定义 'conda' shell 函数
    . "$CONDA_PATH/etc/profile.d/conda.sh"
else
    echo "警告: 未找到 $CONDA_PATH/etc/profile.d/conda.sh。"
    echo "将 Conda bin 目录添加到 PATH 作为备用方案。"
    export PATH="$CONDA_PATH/bin:$PATH"
fi

# 4. 激活 'base' 环境
#    现在 'conda' 函数应该已经定义了
echo "正在激活 'base' 环境..."
conda activate base

echo ""
echo "Conda 设置完成！"
echo " 'base' 环境现在已激活，并且 ~/.bashrc 已配置。"