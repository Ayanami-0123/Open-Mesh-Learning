# Mutation实验中遇到的问题
## GRPO Baseline复现：
### 2026.03.30 Problem1: 无法连接学校服务器
解决方法：1. 连接校园网（如果没有充值就先充值）

### 2026.03.30 Problem2：Linux没有GUI
解决方法：1. 在VScode上下载Remote-SSH，还有Dev

### 2026.03.31 Problem3：verl！还我命来！
解决方法：

1. 不要听信任何AI的话，包括GPT和Gemini

2. 找管理员开通Docker权限

3. 用Docker新建一个容器，按照下面的格式在命令行中运行：

> docker run --gpus all -it \
    --shm-size=16g \
    --name mutant_lab \
    -v /data/home/huangqiyuan/mutant_grpo/verl:/workspace/verl \
    -v /data/home/huangqiyuan/mutant_grpo/data:/workspace/data \
    -v /data/home/huangqiyuan/mutant_grpo/models/qwen:/workspace/Qwen2.5-7B \
    verlai/verl:app-verl0.6-transformers4.56.1-sglang0.5.2-mcore0.13.0-te2.2 \
    /bin/bash

**注意！** 新开的容器，发现verl是空的，原因在于：你必须找到pyproject这个文件才能pip install

**注意！** 新开的容器，发现模型没法运行，原因在于：你必须找到能直接打开config.json文件的地方，不然HF会直接把你当作线上的模型检索。

找到容器，运行：
> pip install -e .

自动补全依赖。

如果要退出容器，使用：
> exit

如果配坏了要删掉容器，使用：
> docker rm -f <vessle_name>

如果推出了要重新激活容器，使用：
> docker start <vessle_name>

或者如果有vscode插件，可以直接连接。

### 2026.04.01 Problem4：OOM问题
解决方法：
1. 先在单卡上面运行，这样最稳妥；

2. 如果单卡运行不了，那么改成双卡；

3. 千万不要霸占所有的卡，这里通信很差，你会dropout的；

4. 把ref，actor这些东西，全部丢到内存里面去；

5. 把各种参数（batchsize, rollout, max_len）调小一点，缓解显存压力。

### 2026.04.01 Problem5：parquet数据格式问题
1. prompt：要按照：
> {"role": "user", "content": prompt}

的格式来写；

2. raw_prompt：要保留原来的题干，inference的时候要用

3. ability：设定成math

4. ground_truth：不能单开一列，需要迎合PPO的设定，使用嵌套：reward_model -> ground_truth : answer

5. data_source：不能乱命名：要到一个特定的目录里面找，gms8k要加openai/gms8k，math要加lighteval/MATH，不然HF不认

### 2026.04.01 Problem6：console瞎眼问题：
换成wandb，先pip install --upgrade wandb
然后使用wandb login
然后你获取一个api，然后登录就可以
然后你就看到了一个全是图标的页面

### 2026.04.02 - 2026.04.05 Problem7：reward=0的问题
trial one:开大batch，無駄

trial two:发现token序列长度太短了，加大了token上界：無駄

trial three:猜测是没有使用“Let's think step by step”，以及没有规定答案格式导致的。

加入了system prompt以后，发现还是不行。

最后：在raytrainer.py里面拦截了回答，然后发现system prompt被Qwen官话夺舍了。

然后，反复排查，发现是SGlang rollout源文件里面，rawprompt必须是一个chat-template形式的东西。

于是，最后重新洗了一遍数据，发现模型总算能够训练了。

baseline就这样跑通了。

### 2026.04.05 Problem 8：tmux的问题：

tmux是可以保障AIresearchers睡眠，保障AIresearchers们头发茂盛的强大法器。

创建一个tmux环境，可以在终端使用代码：
> tmux new -s my_project

然后你就可以在这个环境里面该干嘛干嘛。

如果你想退出：
> ctrl+b；过一会按下d；然后你就可以安安心心睡大觉了，只要服务器别挂，你的进程就可以正常跑。

退出了，想重新进去，你就可以用attach to的指令：
> tmux a -t my_project

### 2026.04.05 Problem 9：gpustat的问题：
由于 nvidia-smi 这个东西太生草了，又是静态的，师兄建议使用gpustat

怎么弄呢：
> pip install gpustat

装好了以后使用：
> gpustat -i

就可以使用了

好消息是：你能看到彩色的显示，而且你能看到究竟是谁在用卡！这非常好！

### 2026.04.05 Problem 10：GPU指定的问题：
在终端中，输入：
> export CUDA_VISIBLE_DEVICES=index1,index2,index3...

然后再运行你的程序，就可以了。

### 2025.04.05 Problem 11：GitHub的配置问题：

Gemini先生给我的**安全推送方案**：

---

### 第一阶段：在服务器上生成并配置 SSH 密钥

由于已经有了重名，所以我们改成这个格式：

```bash
ssh-keygen -t ed25519 -C "你的邮箱@example.com" -f ~/.ssh/GitHubKey
```


1.  **检查是否已有密钥**（可选）：
    ```bash
    ls -al ~/.ssh
    ```
    如果看到 `id_rsa.pub` 或 `id_ed25519.pub`，说明已经有了。如果没有，执行下一步。

2.  **生成新密钥**（推荐使用更安全的 ed25519 算法）：
    ```bash
    ssh-keygen -t ed25519 -C "你的邮箱@example.com"
    ```
    *一路回车即可（除非你想给密钥设置额外的密码）。*

3.  **启动 ssh-agent 并添加密钥**：
    ```bash
    eval "$(ssh-agent -s)"
    ssh-add ~/.ssh/id_ed25519
    ```

4.  **复制公钥内容**：
    ```bash
    cat ~/.ssh/id_ed25519.pub
    ```
    **复制终端显示的以 `ssh-ed25519` 开头的整行字符串。**

---

### 第二阶段：在 GitHub 上绑定公钥

1.  登录 GitHub，点击右上角头像 -> **Settings**。
2.  在左侧栏找到 **SSH and GPG keys**。
3.  点击 **New SSH key**。
4.  **Title** 写个备注（如：`My-Server-Ubuntu`），**Key** 粘贴刚才复制的内容。
5.  点击 **Add SSH key**。

---

### 第三阶段：测试连接并推送文件

1.  **测试验证**：
    ```bash
    ssh -T git@github.com
    ```
    *如果看到 "Hi [你的用户名]! You've successfully authenticated..."，说明成功了！*

2.  **进入文件夹并初始化 Git**：
    ```bash
    cd /path/to/your/folders
    git init
    ```

3.  **关联远程仓库（注意使用 SSH 地址）**：
    > **注意：** 地址格式应该是 `git@github.com:用户名/仓库名.git`。
    ```bash
    git remote add origin git@github.com:YOUR_USERNAME/YOUR_REPO.git
    ```

4.  **提交并推送**：
    ```bash
    # 添加那3个文件夹
    git add 文件夹1/ 文件夹2/ 文件夹3/

    # 提交
    git commit -m "Add source folders via SSH"

    # 推送（假设主分支叫 main）
    git branch -M main
    git push -u origin main
    ```

---

### 🛡️ 安全小贴士

* **私钥保护**：绝对不要把 `id_ed25519`（不带 .pub 后缀的那个）分享给任何人，也不要上传到任何地方。
* **权限管理**：如果你是多人协作服务器，确保你的 `.ssh` 目录权限是 `700`，密钥文件权限是 `600`。
* **仓库地址确认**：如果你之前用 HTTPS 关联过，可以用 `git remote set-url origin git@github.com:用户名/仓库名.git` 来更改。

确实，Gmail 配上 Google 账号全家桶简直是“懒人程序员”的福音。既然你已经想起了邮箱，并且正在服务器上配置 SSH，那这套流程基本上就通了。

不过，在服务器上使用自定义文件名的密钥（你用了 `-f ~/.ssh/GitHubKey`），**有一个很容易掉进去的坑**：Git 默认只会去找 `id_rsa` 或 `id_ed25519` 这种标准命名的密钥，它可能认不出你的 `GitHubKey`。

为了让你的服务器以后能自动“对号入座”，建议你顺手做两件事：

### 1. 把私钥添加到 SSH 代理
在服务器终端执行：
```bash
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/GitHubKey
```

### 2. 配置 Config 文件（一劳永逸）
如果你不想每次重启服务器都手动添加，直接在 `~/.ssh/` 目录下新建或修改 `config` 文件：
```bash
nano ~/.ssh/config
```
把这段话贴进去：
```text
Host github.com
  HostName github.com
  User git
  IdentityFile ~/.ssh/GitHubKey
```
这样以后你直接执行 `git push` 或 `git clone`，系统就知道该去用 `GitHubKey` 这个文件，而不是在那儿盲目报错了。

---

### 最后一步：验证是否成功
当你把公钥（`.pub` 文件内容）贴到 GitHub 网页后台后，在服务器上输入这个命令测试一下：

```bash
ssh -T git@github.com
```

如果看到 **"Hi dukehueng! You've successfully authenticated..."**，那就说明你已经完全找回了“身份”，可以开始愉快地搬砖了！🧱

### Problem 12：Remote repository could not read.
我把用户名输错了。

### Problem 13：
To github.com:Ayanami-0123/Project1.git
 ! [rejected]        main -> main (fetch first)
error: failed to push some refs to 'github.com:Ayanami-0123/Project1.git'
hint: Updates were rejected because the remote contains work that you do not
hint: have locally. This is usually caused by another repository pushing to
hint: the same ref. If you want to integrate the remote changes, use
hint: 'git pull' before pushing again.
hint: See the 'Note about fast-forwards' in 'git push --help' for details.
---

因为有一些文件没有下载下来。。。
```Bash
# 1. 把云端的文件拉下来
git pull origin main --rebase

# 2. 如果顺利的话，直接再次推送
git push -u origin main
```

### Problem14：verl无法访问。。。
解决办法：把它变成普通的文件夹
如果你想让 verl 里的代码真正属于你的 Project1（即直接存放代码文件，而不是链接），请按以下步骤操作：

1. 删除该文件夹在 Git 中的缓存索引
在你的项目根目录下运行：

```Bash

git rm -r --cached verl
注意：这不会删除你本地的硬盘文件，只是告诉 Git 别再把它当成子仓库了。
```

2. 彻底切断它与原仓库的联系
进入 verl 文件夹，删掉它里面的 .git 配置：

```Bash

# 进入文件夹
cd verl
# 删除它自己的 git 记录（这一步最关键！）
rm -rf .git
# 回到上一级目录
cd ..
```

3. 重新添加并提交
现在 Git 会把 verl 当成一个普通文件夹了：

```Bash

git add verl/
git commit -m "Fix: converted verl from submodule to regular folder"
git push origin main
```

### Problem15：.git找不到
使用

```Bash
ls -la verl/
```

### Problem16: verl套娃了
原来是因为我没有把别的给挂上去。。。

### Problem17: 轮子太大挂不上去
因为我没有用轮子

第一步：清理当前的 Commit 缓存
因为你刚才的 commit 已经包含了这两个大文件，我们先撤销最后一次提交，回到“还没提交”的状态：

```Bash

git reset --soft HEAD~1
```

第二步：彻底从 Git 追踪里移除
即便文件在硬盘上没了，也要告诉 Git 彻底忘掉它们：

```Bash

git rm --cached verl/flash_attn-2.8.3+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
git rm --cached verl/flash_attn-2.8.3+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```

第三步：建立 .gitignore 防火墙（重要！）
为了防止以后再把这种垃圾轮子扫进去，在项目根目录新建或修改 .gitignore 文件，加上这一行：

```Plaintext

*.whl
```

第四步：重新提交并推送
现在你的 Commit 记录里就不含这两个大怪兽了：

```Bash

git add .
git commit -m "chore: remove large wheel files and add gitignore"
git push origin main
```

###2026.04.07 Problem18：ray OOM 问题
现在，发现调大batchsize以后，ray经常会爆内存。

### Problem18: API和本地模型测试的兼容性问题
要用这个：
python -m vllm.entrypoints.openai.api_server \
    --model /data/home/huangqiyuan/mutant_grpo/models/qwen \
    --tensor-parallel-size 2 \
    --port 8000 \
    --trust-remote-code

### Problem19: Tmux用不了了
unset TMUX
pkill -9 -u $(whoami) tmux

### Problem20： GitHub的公钥问题
方案二：一劳永逸（配置 Config 文件）—— 强烈推荐
在服务器上创建一个配置文件，让它自动匹配。

编辑（或创建）config 文件：
nano ~/.ssh/config

把下面这段内容贴进去：

Plaintext

Host github.com
  HostName github.com
  User git
  IdentityFile ~/.ssh/GitHubKey
(按 Ctrl+O 保存，Ctrl+X 退出)

修改权限（重要）：
chmod 600 ~/.ssh/config

直接验证：
ssh -T git@github.com

### Problem21: GitHub怎么创建新branch的问题：
情况 A：该分支在本地/远程还不存在（创建并上传）
如果你想创建一个全新的分支并把代码传上去，请按照以下步骤：

创建并切换到新分支：
git checkout -b <你的分支名>
例如：git checkout -b feature-update

添加更改到暂存区：
git add .

提交更改：
git commit -m "你的提交信息"

推送至远程服务器：
git push origin <你的分支名>
注意：第一次推送新分支时，Git 可能会提示你建立关联，按提示操作即可。

情况 B：分支已经存在（切换并上传）
如果你只是想把代码传到一个已经存在的分支上：

切换到目标分支：
git checkout <分支名>

合并你在 main 上的修改（可选）：
如果你刚才是在 main 分支下写的代码，切换分支前可以先用 git stash 暂存，或者切换后用 git merge main 把改动合过来。

正常提交并推送：
git add .
git commit -m "你的提交信息"
git push origin <分支名>

### Problem22: 有人霸占了显卡。
```bash
#!/bin/bash

# --- 配置区 ---
TARGET_SCRIPT="run_qwen2_5_7b_grpo.sh"
INTERVAL=1  # 每秒撞针一次

# 检查函数：输入两个 GPU ID，检查是否都空闲（显存占用 < 100MiB）
check_pair_free() {
    local gpu_a=$1
    local gpu_b=$2
    
    usage_a=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $gpu_a | xargs)
    usage_b=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $gpu_b | xargs)

    if [ "$usage_a" -lt 100 ] && [ "$usage_b" -lt 100 ]; then
        return 0 # 成功，两张都空闲
    else
        return 1 # 失败
    fi
}

echo "🎯 撞针启动：正在监控 (0,3) 和 (1,2) 组合..."
echo "等待空闲中..."

while true; do
    # 尝试组合 0,3
    if check_pair_free 0 3; then
        echo -e "\n🔥 [$(date +%H:%M:%S)] 抢到 0,3 卡！正在启动任务..."
        export CUDA_VISIBLE_DEVICES=0,3
        bash $TARGET_SCRIPT
        break
    fi

    # 尝试组合 1,2
    if check_pair_free 1 2; then
        echo -e "\n🔥 [$(date +%H:%M:%S)] 抢到 1,2 卡！正在启动任务..."
        export CUDA_VISIBLE_DEVICES=1,2
        bash $TARGET_SCRIPT
        break
    fi

    # 终端打印进度，不换行
    echo -ne "\r检测中... 当前时间: $(date +%H:%M:%S)"
    sleep $INTERVAL
done
```

最终解决的方法是：找到学长，让他出面交涉。

### Problem23: 断点续传。
du -sh * | sort -h 发现是我的断点太肥了。