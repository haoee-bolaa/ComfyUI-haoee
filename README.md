# 网站：[https://www.haoee.com/maas/services](https://www.haoee.com/maas/services)
# ****好易智算——ComfyUI工作流调用api说明****

## ****第一步：在好易智算，申请API Key及账户充值****

1.  打开好易智算：[<u>https://www.haoee.com/maas/services?ic=VUZUlOJo</u>](https://www.haoee.com/maas/services?ic=VUZUlOJo)，注册或登录
2.	右上角确认账号里面有余额或充值
3.  右上角点击工作台，然后在左侧菜单找到算力调用api

  <img width="1572" height="414" alt="fc52e76fe4b324eba05ff524c16af607" src="https://github.com/user-attachments/assets/c8d12bf0-45d3-44ba-9993-af82aa49f26b" />


4.  进入以后复制API Key即可

  <img width="1584" height="465" alt="5c68bfe4840bf12d7746aef368266d2e" src="https://github.com/user-attachments/assets/895240ad-4ed0-4a4b-986e-1f41585ffb1f" />


## ****第二步:在ComfyUI工作台使用密钥****

5.  请点击下载安装插件：https://github.com/haoee-bolaa/ComfyUI-haoee

6.  进入工作流市场，选择需要使用的API模型对应的工作流（以Gemini image为例）

7.  打开工作流以后，找到“好易API key“这个节点，把上面复制的key填入

<img width="1362" height="810" alt="b2f381815e43c1df1e79a27f226b0d56" src="https://github.com/user-attachments/assets/4773aec6-a322-49a3-8822-0cd2f35c8a0a" />


7.  好易API key账号就可以在工作流里面进行扣费了

## ****第三步：费用说明****

8.  运行带有API key的工作流费用包含：运行工作流调度的算力费用+API的调用费用

9.  工作流调度的算力费用对应：ComfyUI云端工作台的账户：[https://cl.haoee.com/](https://cl.haoee.com/)，按工作流运行时长扣费，大约0.002分/秒（——本地忽略此费用）

10.  API调用的费用对应的账户：[<u>https://www.haoee.com/maas/services?ic=VUZUlOJo</u>](https://www.haoee.com/maas/services?ic=VUZUlOJo)，具体调用模型调用价格请查看：好易智算Maas模型服务板块

11.  现目前在ComfyUI里面，API key调用好易Maas模型，以正式渠道价格的为准，确保用户工作流运行的稳定性

12.  以运行一次：gemini-image 工作流 2K图片为例，消耗的费用为（——本地忽略此费用）： 

运行工作流时长（0.002分/秒 x 4秒 = 0.008分）

\+ 模型调用（gemini-3-pro-image-preview 2K：0.495元） = 0.503元
 
<img width="1590" height="396" alt="d3847520dd93d8e83eeca7b68c8bbf1f" src="https://github.com/user-attachments/assets/95596371-8aec-4d03-b9b1-1bd088092dd8" />
<img width="900" height="806" alt="2d008ddaf8e0b6515f247cb02a35afaa" src="https://github.com/user-attachments/assets/c17c7632-e83a-4bb3-beec-a709026f69ec" />
