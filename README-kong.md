在项目根目录执行pip install -e .指令，将当前文件代码中引用的ultralytics链接挂载到本地源代码而不是官方ultralytics库中，这样可以使得自己在源代码修改的内容生效

# git上传文件事项

~~~bash
#去除仓库链接
git remote remove origin

#重新绑定仓库链接
git remote add origin https://github.com/你的用户名/my-yolov8-road-surveillance.git

#核对远程仓库链接
git remote -v

#上传对应的文件
git add ......

#添加注释
git commit -m "这是注释"

#推送更新
git push origin main
~~~

