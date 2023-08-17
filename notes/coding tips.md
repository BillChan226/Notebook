### Python

#### torch

1. torch tensor需要关注矩阵维度的表示方法:

+ 一个[A, B, C, D]的四维矩阵，第一个list里应该有A个element；第二个list有B个element以此类推；例如对于torch.arange(24).view(1,2,3,4)，输出应为：

  ![image-20230805175141844](https://raw.githubusercontent.com/BillChan226/Notebook/main/image/image-20230805175141844.png)

  将每个最内层的4维数组视作$m_i$, 则这个tensor应具备以下结构：tensor = [ [ [$m_1$, $m_2$, $m_3$], [$m_4$, $m_5$, $m_6$] ] ]

  即最外层的数组是1维；第二层2维；第三层3维；第四层4维

+ 以此类推，在tensor的表示方式中，若一个7行8列的tensor.shape=[7, 8]，说明其最外层有7个element，每个element包含一个8维的list。

+ 对于tensor.view()和tensor.reshape()函数而言，其**仅改变括号插入的位置，不改变element的排列顺序**！然而，tensor.T取转置会改变element的排列顺序。因此对于tensor.arange(24).view(4,6) 不等于tensor.arange(24).view(6,4).T:

  + tensor.arange(24).view(4,6):

  ![image-20230805181218531](https://raw.githubusercontent.com/BillChan226/Notebook/main/image/image-20230805181218531.png)

  + tensor.arange(24).view(6,4).T:

  ![image-20230805181303648](/Users/zhaorunchen/Library/Application Support/typora-user-images/image-20230805181303648.png)

2. torch.exp和torch.exp都是element-wise运算。
3. torch.sum需要指定axis, axis取第几个（从0开始）就在第几层求和。例如对于[7, 8]的一个tensor，axis=0则指定其对列求和，得到一个8维tensor；axis=0则指定其对行求和（每个内部list求和），得到一个7维tensor。



### Linux

+ 保持ssh断连重连后终端仍然在线，安装screen

  ```
  brew (apt-get) install screen
  ```

+ 检查screen是否安装好

  ```
  screen -v
  ```

+ 创建screen窗口

  ```
  screen -S screen_name
  ```

+ 主动detach窗口

  ```
  screen -d screen_name
  ```

+ 恢复screen窗口

  ```
  screen -r screen_name
  ```

  