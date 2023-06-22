# 自然语言处理基础 - Dependency Parsing 作业报告

## 文件结构与运行说明

提交的压缩包解压后应有如下结构

```
├─ report.pdf
├─ requirement.txt
├─ prediction.json
├─ main.py
├─ parser_transitions.py
├─ parser_utils.py
├─ parsing_model.py
├─ predict.py
├─ trainer.py
├─ utils.py
├─ data
|   ├─ preprocessing.py
|   ├─ newglove.6B.100d.txt
|   ├─ dev.conll
|   ├─ test.conll
|   └─ train.conll
├─ results
|   └─ model.weights
```

`./data/newglove.6B.100d.txt`是预训练的词向量模型，可在[https://nlp.stanford.edu/projects/glove/](https://nlp.stanford.edu/projects/glove/)下载。实际上为防止所需提交的文件过大，对其进行了预处理，只保留了训练数据中出现的单词及其词向量，预处理的代码在`./data/preprocessing.py`。

运行`python main.py`将进行模型训练（预测依存边关系），将`parser_utils.py`中`Config.unlabeled`参数设置为`True`后运行`python main.py`则进行只预测head的训练。

运行`python predict.py`将使用`./results/las.weights`对`./data/test.conll`中的数据进行测试并打印UAS和LAS，同时保存预测的结果到`./prediction.json`，文件结构为一个列表，里面的每一个元素是一个列表，表示一个句子的预测结果，这个结果内的元素使用`(head, label)`表示的parsing。例如，若设整体列表为`lst`，则`lst[3][7] = (5, 'amod')`表示第3句话有第5个词指向第7个词的`'amod'`依存关系。

## 代码分析

## `parser_utils.py`

#### 1. `Parser.__init__`

对给定数据集构建字典`tok2id`和`id2tok`，实现parsing labels、POS tags、words从字符串到整数的映射和逆映射。
对操作构建字典`trans2id`和`id2trans`，实现parsing操作到整数的映射和逆映射。

#### 2. `Parser.vectorize`

将给定的一组examples向量化为列表。列表中每一个元素是一个字典，存储了相应句子每个单词的word、POS tag、parsing head、parsing label。

#### 3. `Parser.extract_features`（自实现）

为一个特定的状态ex提取特征，参考文章[https://aclanthology.org/D14-1082/](https://aclanthology.org/D14-1082/)，提取的特征包括：

- 词特征：
  + (1) stack和buffer顶端的三个词s1, s2, s3, b1, b2, b3
  + (2) stack顶端两个词最左、次左、最右、次右的孩子lc1(si), rc1(si), lc2(si), rc2(si), i = 1, 2
  + (3) stack顶端两个词最左孩子的最左孩子、最右孩子的最右孩子lc1(lc1(si)), rc1(rc1(si)), i = 1, 2
  + 所有不存在的词用NULL代替，词特征共18维
- POS特征：词特征中所有词的POS tag，不存在的用P_NULL代替，共18维
- label特征：词特征(2)(3)中单词与孩子间的parsing label，不存在的用L_NULL代替，共12维
  
按照`config`中的`use_pos`和`use_dep`的值确定是否采用POS特征和label特征。
最后将提出的特征利用`tok2id`（实际上是用ex中存储的列表）转化为整数后返回。

#### 4. `Parser.get_oracle`（自实现）

根据当前状态，返回应该执行的操作编号。
取出stack顶端的两个词（若stack长度不足，根据buf中是否还有元素确定返回`SHIFT`或`None`）`x0`和`x1`以及它们的head和label
- 若`head(x1) == x0`，说明存在`x0`指向`x1`的一条关系为`label(x1)`依存关系，执行`LEFT-ARC`
- 若`head(x0) == x1`，说明存在`x1`指向`x0`的一条关系为`label(x0)`的依存关系，假如buf中没有其他以`x0`为head的元素，执行`RIGHT-ARC`
- 否则根据buf中是否还有元素确定返回`SHIFT`或`None`
  
#### 5. `Parser.create_instances`（有自实现部分）

为给定的examples生成action序列。
对每一个example，初始化stack、buf、arc，反复使用`Parser.get_oracle`获取操作并更新stack、buf、arc，直至获取的操作为None。
更新的过程为自实现部分，与`parser_transitions.py`中`PartialParse.parse_step`的实现方式完全一致，参见后文中该函数部分的报告。

#### 6. `Parser.legal_labels`

给定stack和buf，返回所有合理的操作，用于在`Parser.create_instances`中判断获取到的操作的合理性。

#### 7. `Parser.parse`

对输入的数据进行依存关系的预测。预测的具体流程在`parser_transitions.py`中的`minibatch_parse`函数中实现。
由于在`(utils.py) evaluate`函数中实现了计算UAS和LAS的功能，故本函数中原有的计算UAS的代码被删去。

#### 8. `ModelWrapper.__init__`

初始化ModelWrapper，输入包括一个Parser和待预测的数据集（及其元素从地址到索引的字典）。

#### 9. `ModelWrapper.predict`

输入一组已完成部分parsing的句子，利用这部分已完成的parsing提取特征输入模型，在所有合法的parsing label中选择得分最高的一项作为下一步parser的动作。

#### 10. `read_conll`

读取`conll`文件，将数据集中的每个句子保存为一个关键字为`'word', 'pos', 'head', 'label'`的字典。各个键的值是等长的四个列表。

#### 11. `build_dict`

输入`keys`，构建`keys`中的元素到整数的字典映射，若同时输入`n_max`，则字典的键为`keys`中出现频率最高的前`n_max`个元素。

#### 12. `load_and_preprocess_data`

读取训练集、验证集、测试集，利用训练集构建parser并利用`Parser.create_instances`生成相应的动作序列。
**注** 由于在模型中使用了预训练的词向量，因此在函数中也添加了读取词向量并返回的功能。

## `parser_transitions.py`

#### 1. `PartialParse.__init__`

初始化`PartialParser`，`stack`初始时为`['ROOT']`，`buffer`初始时为句子中的所有单词，`dependencies`为空。

#### 2. `PartialParse.parse_step`

对于给定的一个`transition`，更新`stack`、`buf`和`dependencies`。

- `transition == SHIFT`，取出`buf`顶部元素加入`stack`
- `transition == LEFT-ARC`，建立`stack`顶部第一个元素`s1`指向第二个元素`s2`的关系并加入`dependencies`中，而后移除`s2`
- `transition == RIGHT-ARC`，建立`stack`顶部第二个元素`s2`指向第一个元素`s1`的关系并加入`dependencies`中，而后移除`s1`

#### 3. `minibatch_parse`

对给定的一组句子和`batch size`进行`parsing`。
为每一个句子单独建立`PartialParse`的示例，每次选取`batch size`个未完成parsing的句子，利用`ModelWrapper.predict`预测下一步parsing的动作并更新相应的`PartialParse`，直至完成所有给定句子的依赖关系预测。

## `paring_model.py`

#### `ParsingModel.__init__`

初始化模型的参数列表如下

- `embeddings`：预训练的词向量，实验中使用[glove.6B.100d](https://nlp.stanford.edu/projects/glove/)
- `n_features`：特征维数，利用`Parser.extract_features`和`Config`中是否使用label和POS的相关设置确定
- `hidden_size`：隐层神经元数，默认为256
- `n_classes`：预测的种类数，若不预测label则为3，否则为`1 + 2 * deprel`
- `dropout`：dropout层的概率，默认为0.5

#### `ParsingModel.forward`

使用的模型为单隐层神经网络，架构为（测试时不使用Dropout）
`
input features --(Embedding)--> dense features --(Linear)--(ReLU)--(Dropout)--> hidden --(Linear)--> logits
`

## `trainer.py`和`main.py`

在`main.py`中制定损失为交叉熵，优化器为Adam。

在`trainer.py`中，利用模型预测得到`logits`，与GT计算交叉熵损失并做反向传播更新参数。
此外，为实现batch的训练，还定义了以下函数，输入为数据和`batch_size`，并返回一个迭代器

#### `get_batch`

输入数据和`batch size`，构建等于数据长度的idx索引并shuffle，用for循环遍历数据，每一轮yield一个`batch size`大小的数据。

## LAS代码改动

#### `(parser_utils.py) ModelWrapper.predict`
改动部分：将源代码中的
```
pred = ["S" if p == 2 else ("LA" if p == 0 else "RA") for p in pred]
```
改为
```
action_pred = ["S" if p == self.parser.n_trans - 1 else ("LA" if 0 <= p < self.parser.n_deprel else "RA") for p in pred]
```
并同时返回`pred`和`action_pred`，使模型能够同时预测parsing动作和标签。

#### `(parser_transitions.py) minibatch_parse`

在原有parsing过程中存储`(head, dependency)`对的基础上，额外存储模型所预测的关系。

#### `(utils.py) evaluate`

修改了函数接口，使得评估UAS和LAS的方式与训练代码中的数据格式相适应。

## 实验结果

在测试集上的UAS和LAS分别为87.25%和85.15%。

以下是一个句子的parsing结果。
原句
``The 49 stock specialist firms on the Big Board floor -- the buyers and sellers of last resort who ``
``were criticized after the 1987 crash -- once again could n't handle the selling pressure .``
Ground Truth
``(5, 'det') (5, 'nummod') (5, 'compound') (5, 'compound') (31, 'nsubj') (10, 'case') (10, 'det') ``
``(10, 'compound') (10, 'compound') (5, 'nmod') (5, 'punct') (13, 'det') (5, 'dep') (13, 'cc')``
``(13, 'conj') (18, 'case') (18, 'amod') (13, 'nmod') (21, 'nsubjpass') (21, 'auxpass') (13, 'acl:relcl')``
``(25, 'case') (25, 'det') (25, 'nummod') (21, 'nmod') (5, 'punct') (28, 'advmod') (31, 'advmod') (31, 'aux') ``
``(31, 'neg') (0, 'root') (34, 'det') (34, 'compound') (31, 'dobj') (31, 'punct')``
预测结果
``(5, 'det') (5, 'nummod') (5, 'compound') (5, 'compound') (0, 'root') (10, 'case') (10, 'det') ``
``(10, 'compound') (10, 'compound') (5, 'nmod') (5, 'punct') (13, 'det') (31, 'nsubj') (13, 'cc') ``
``(13, 'conj') (18, 'case') (18, 'amod') (13, 'nmod') (21, 'nsubjpass') (21, 'auxpass') (18, 'acl:relcl') ``
``(25, 'case') (25, 'det') (25, 'nummod') (21, 'nmod') (31, 'punct') (28, 'advmod') (31, 'advmod') (31, 'aux') ``
``(31, 'neg') (5, 'dep') (34, 'det') (34, 'compound') (31, 'dobj') (5, 'punct')``