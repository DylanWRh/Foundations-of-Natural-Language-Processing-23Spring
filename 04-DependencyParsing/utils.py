from tqdm import tqdm

def evaluate(preds, rels, golds, deprel=39):
    #### pred 是一个列表，其中每个元素是一个句子的parsing结果
    #### 该结果也用一个列表表示，第i个元素是该句中第i个词的预测结果，用(head_idx, rel)表示
    #### 其中head_idx 表示第i个词预测的dependency head的id，rel表示预测的关系类型
    #### gold 和 pred 表示方式相同，将预测结果替换成了gold label
    #### 
    #### 有两种评测指标，uas 和 las
    #### uas 只评价预测的head token idx 是否正确
    #### las 在head token预测正确的前提下，还要求对应的边的关系，也需要预测正确

    # preds: [[(head, dependencies) * len(words)] * len(sentences)]
    # rels:  [[rels * len(words)] * len(sentences)]
    # golds: [{'word': [...], 'label': [...], 'head': [...], 'pos': [...]} * len(sentences)]

    uas_hit, uas_miss = 0, 0
    las_hit, las_miss = 0, 0

    with tqdm(total=len(golds)) as prog:
        for i, ex in enumerate(golds):
            head = [-1] * len(ex['word'])
            label = [-1] * len(ex['label'])
            for (h, t), r in zip(preds[i], rels[i]):
                head[t] = h
                label[t] = r
            for pred_h, gold_h, pred_l, gold_l in zip(head[1:], ex['head'][1:], label[1:], ex['label'][1:]):
                if pred_h == gold_h:
                    uas_hit += 1
                else:
                    uas_miss += 1
                if pred_h == gold_h and (pred_l % deprel) == gold_l:
                    las_hit += 1
                else:
                    las_miss += 1
            prog.update(i + 1)

    # uas_hit, uas_miss = 0, 0
    # las_hit, las_miss = 0, 0
    # for sent_pred, sent_gold in zip(preds, golds):
    #     for pred, gold in zip(sent_pred, sent_gold):
    #         if pred[0] == gold[0]:
    #             uas_hit += 1
    #         else:
    #             uas_miss += 1
    #         if (pred[0] == gold[0]) and (pred[1] == gold[1]):
    #             las_hit += 1
    #         else:
    #             las_miss += 1

    print("The number of UAS right is "+ str(uas_hit))
    print("The number of UAS wrong is "+ str(uas_miss))
    print("UAS is "+str(uas_hit / (uas_hit + uas_miss)))

    print("The number of LAS right is "+ str(las_hit))
    print("The number of LAS wrong is "+ str(las_miss))
    print("LAS is "+str(las_hit / (las_hit + las_miss)))

    return uas_hit / (uas_hit + uas_miss), las_hit / (las_hit + las_miss)


