## BERT面试QA库

### 1. BERT和Transformer中的位置编码的区别？
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Transformer中的position embedding是有sin/cos函数生成的固定值, 该embedding只能标记位置，
但不能标记这个位置有什么用。BERT中的position embedding是和词嵌入类似的，随机生成可训练的Embedding，
不仅可以标记位置，还可以学习到这个位置有什么用。
### 2. BERT和AL-BERT的区别？
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;AL-BERT是在BERT的基础上进行改进的。它设计了参数减少的方法，
用来降低内存的消耗，同时提升模型训练速度。对比原生的BERT模型，AL-BERT主要做了三个优化。\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**1**. 加入了词嵌入的因式分解，将词嵌入矩阵分解成为了两个更小的矩阵。即: V * H -> V * E + E * H,其中V为词表，H为隐层大小，E为embedding大小。\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**2**. 交叉层的参数共享，使得AL-BERT中层与层之间收敛，而BERT系列的模型一般层与层通常是震荡的。\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**3**. 自监督损失函数，BERT中一个训练目标是next sentence prediction(NSP)。NSP是一个二分类任务，主要包括主题预测和连贯性预测。
AL-BERT中引入了sentence order prediction(SOP)。避免主题预测而关注构建句子之间的连贯性。SOP能一定程度上解决NSP任务中不可靠的问题。
### 3. BERT的网络结构以及训练过程？
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;BERT是用了Transformer中encoder的网络，使用大规模的数据进行预训练，主要两个子任务: 一个是mask ML，\
利用上下文随机预测这些字。一个是预测两个句子是否为上下句。
### 4. BERT中mask LM 中mask如何实施的?
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;对输入句子中的词以15%的概率进行mask。\
mask的方式包含三种:分别为80%的概率用[mask]替换，以10%的概率替换为一个其他字，10%的概率不替换。
### 5. multi-head attention的具体结构？
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Attention是把一个query，一个key-value\
集合映射成一个输出。其中query,key,value和输出都是向量。输出是value的加权求和，表示的是query\
和key的相关程度。即Attention表示的是查询(query)到一个系列(key-value)对的映射。
multi-head attention表示的是查询到一个系列对在多个表征子空间的映射的加权和。
### 6. BERT中的embedding包括哪几个部分?
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;word emb, pos emb还有sentence emb。\
其中pos emb和word emd类似，可以进行自适应学习。
### 7. BERT采用那种Normalization结构，LayerNorm和BatchNorm的区别，LayerNorm结构的参数作用?
BERT采用的是LayerNorm结构，LayerNorm和BatchNorm的区别是在于做归一化的维度不同。BatchNorm\
针对一个batch里面的数据进行归一化，针对的单个神经元。layerNorm针对的单个样本，不依赖于其他数据。\
LayerNorm主要有w和b两个维度的参数，其作用是对本网络的输出进行额外的约束，将其限定至指定的区间。
### 8. Transformer attention公式中的根号d的作用？
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Q和K点积后的结果大小是跟维度成正比的，所以经过\
softmax以后，梯度会变得很小，除以根号d可以使得attention的权重分布方差为1，而不是dk，解决了梯度\
消失问题。这里的Q和K都是dk维度的向量
### 9. BERT中使用的wordpiece的作用?
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;其核心思想是将单词打散为字符，然后根据片段的组合\
频率,最后将词切成片段处理，极大的降低OOV问题。如: cosplayer, 用分词容易出现UNK，但利用BPE\
可以将其切分成cos play er，模型可以根据词根和前缀等信息学习到这个词的大致信息。wordpiece和BPE类似,\
每次从词表中选出两个子词合并成新的子词。
### 10. BERT中的前馈神经网络FFN的作用?
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;FFN主要提供非线性变换的作用。由于self-attention中计算\
属于线性变换。通过拼接后经过全连接层仍无激活函数，所以multi-head attention层都是没有经过非线性变换的。在其\
后的FFN，包括2个全连接层，其中第一个全连接层含有激活函数。
### 11. GELU的原理，对比RELU有何缺点？
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;ReLU会确定性的乘以一个0或者1(x<0时乘以0, 否则乘以1)，GeLU\
虽然也是将输入乘以0或者1，但是输入到底乘以0还是1，取决于输入自身情况下随机选择。GeLU的优点就是在ReLU上增加随机\
因素，x越小越容易被mask掉。
### 12. 为什么用LN，不用BN？
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;BN是为每一个小batch计算每一层的平均值和方差，LN独立计算梅一层\
每一个样本的均值和方差，对于RNN系列的模型而言，sequence的长度不一致，存在很多padding表示无意义的信息，如果用BN\
会导致embedding损失信息。LN是在隐层大小的维度进行的，和batch及seq_len无关，每个隐层计算自己的均值和方差。
### 13. LN在BERT的哪个位置?
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;embedding层结束，FFN层结束
### 14. BERT中LN归一化作用
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;LN在BERT中起白化的作用，增强模型的稳定性，如果删除会导致模型无法收敛
### 15. 残差连接作用
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Transformer堆叠很多层，减缓梯度消失。
### 16. 为什么用双线性点积模型，即Q和K两个向量
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;双线性点积模型使用Q和K两个向量，而不是只用一个Q向量，这样引入非对\
称性，更具鲁棒性(attention对角元素值不一定为最大，即当前位置对自身对得分不一定最高)。
### 17. Transformer中的非线性来自于哪个模块？
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;FFN中的gelu激活函数和self-attention(softmax)
### 18. 学习率warn-up策略
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;针对学习率的一种优化方式，它在训练开始时使用一个相对较小的学习率。\
训练了一些epoches或者steps(如: 4个epoches, 10000steps等)，再修改为预先设置的学习率来进行训练。
原因是刚开始训练时，模型的权重参数是随机初始化的，若选择一个较大的学习率，可能带来模型的不稳定，采用warn-up策略。\
可以使得开始训练的epoches利用较小的学习率预热，模型慢慢趋于稳定后，再使用预先设置的学习率进行训练，加快模块收敛速度。\
### 19. BERT的有点和缺点
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;优点: 1.考虑到双向上下文信息，2.可以得到动态的词向量和句子向量\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;缺点: 1.在预训练的时候mask，在fine-tuning的时候没有mask，训练\
数据和测试数据不匹配，会存在一定的误差。2.缺乏生成能力。3.对mask的tokens没有考虑相关性，即假设词之前是独立的。4.出\
现OOV的情况。



