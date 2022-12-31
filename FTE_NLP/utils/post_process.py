import nltk

def postprocess_text(generated, labels):
    """
    change the formate of generated strings
    :param generated:
    :param labels:
    :return:
    """
    generated = [" ".join(gen.split()[1:]) for gen in generated]
    labels = [" ".join(label.split()) for label in labels]

    generated = ["\n".join(nltk.sent_tokenize(gen)) for gen in generated]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]
    return generated, labels


def postprocess_event(preds, labels, ids2tags):
    labels_tag = list()
    for label in labels:
        temp_holder = list()
        for item in label:
            if item != -100:
                item_str = str(item.item())
                temp_holder.append(ids2tags[item_str])
        labels_tag.append(temp_holder)

    preds_tag = list()
    for pred, label in zip(preds, labels):
        temp_holder = list()
        for item, elem in zip(pred, label):
            if elem != -100:
                item_str = str(item.item())
                temp_holder.append(ids2tags[item_str])
        preds_tag.append(temp_holder)

    return preds_tag, labels_tag