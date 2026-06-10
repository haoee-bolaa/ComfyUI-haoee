# 网站：[https://www.haoee.com/maas/services](https://www.haoee.com/maas/services)

## 更新日志
- 2026年6月10日：《好易 LLM》节点添加：deepseek-v4-pro、deepseek-v4-flash、glm5.1模型，移除：deepseek-R1、deepseek-v3.2、GLM-4.7、GLM-4.7-Flash模型
- 2026年6月10日：修正《好易 视频 Doubao》节点，并移除doubao-seedance-1-0-lite-i2v-250428模型
- 2026年6月10日：修正《好易 视频 Wan》节点，添加：wan2.7-i2v模型，移除：wan2.6-i2v、wan2.6-i2v-flash模型
- 2026年6月10日：修正《好易 视频 MiniMax》节点
- 2026年6月9日：移除《好易 绘图 GPT Image》节点，请改用《好易 绘图 GPT Image2》系列节点
- 2026年6月9日：《好易 视频 Kling》（支持 kling-video-o1 / kling-v3-omni）
- 2026年6月9日：增加《好易 视频 Kling v3》
- 2027年6月9日：修正《好易 绘图 GPT Image2（按token）》《好易 绘图 GPT Image2 2K（按次）》
- 2026年6月9日：《好易 绘图 Doubao Seedream》节点增加：doubao-seedream-5-0-260128 模型
- 2026年6月9日：《好易 LLM》节点增加：doubao-seed-2-0-lite-260215、gemini-3.1-pro-preview、gemini-3.5-flash模型
- 2026年6月9日：《好易 LLM GPT5.4》节点改名为《好易 LLM GPT5》，并增加gpt-5.5模型
- 2026年6月9日：移除《好易 LLM GPT》节点
- 2026年6月9日：增加《好易 LLM Gemini》节点
- 2026年6月8日：《好易 绘图 GPT Image2》节点修改为《好易 绘图 GPT Image2（按token）》，支持1k、2k、4k
- 2026年6月8日：新增《好易 绘图 GPT Image2 2K（按次）》节点，支持1k、2k
- 2026年6月8日：新增《好易 绘图 GPT Image2 4K（按次）》节点，支持1k、2k、4k
- 2026年6月8日：节点《好易 绘图 Gemini》新增 gemini-3-pro-image-preview-lite、gemini-3.1-flash-image-preview-lite 模型；
- 2026年6月4日：上线《好易 绘图 GPT Image2》节点；取消《好易 绘图 GPT Image2 图片生成(测试渠道)》节点
- 2026年5月14日：增加《好易 绘图 GPT Image2 VIP》节点
- 2026年5月14日：修正《好易 视频 Sora2》节点，并移除sora-2-pro模型
- 2026年4月28日：增加《好易 视频 Seedance》节点，包含模型Seedance-2-0、Seedance-2-0-fast
- 2026年4月24日：增加《好易 LLM GPT5.4》节点，包含gpt-5.4、gpt-5.4-pro模型
- 2026年4月24日：增加 gpt-image-2 测试渠道节点-《好易 绘图 GPT Image2 图片生成(测试渠道)》
- 2026年4月16日：《好易 绘图 Midjourney》节点增加参数错误提示
- 2026年4月16日：《好易 视频 Wan》节点修改wan2.6-i2v模型强制使用有声
- 2026年4月16日：《好易 视频 Kling》节点BUG修改


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
