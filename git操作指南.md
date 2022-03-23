## git版本控制工具
git init<br/>
在项目文件夹路径下，此命令可以将该项目纳入git版本控制

git config --global user.name '****'<br/>
git config --global user.email '*********'<br/>
开发者信息注册登记

git branch<br/>
查看当前分支

git branch 分支名<br/>
新建分支

git checkout 分支名<br/>
切换建分支

git checkout -b 分支名<br/>
新建并切换到该分支

git branch -d 分支名<br/>
删除分支

git branch -D 分支名<br/>
强制删除分支

git status<br/>
查看当前分支的状态

git add 文件名<br/>
提交文件到缓存区<br>
git commit -m '****'<br/>
提交文件并注释

git commit -am '*****'<br/>
git add 与 git commit的合并操作

git checkout 文件名<br/>
将未提交到暂存区的文件恢复之前未改动的版本

git log --oneline<br/>
查看提交日志，以行的形式显示

git diff<br/>
比较差异

git diff -- staged