import re
import json
from collections import Counter


def load_texttag_file(texttag_filename):
    try:
        with open(texttag_filename, "r") as data_file:
            data_all = data_file.read()
            tags_all = list()
            texts_selected = list()
            tags_selected = list()

            for line in re.split(r'\n\t?\n', data_all):
                if len(line) != 0:
                    texts_line = list()
                    tags_line = list()
                    for item in line.split("\n"):
                        if len(item)!=0:
                            text, tag = item.split("\t")
                            if re.search(r"[@|?|!+?|:|(|)]|\\|\.*?\|-|/|/|/.*?/|http\S+|www\S+", text) == None:
                                texts_line.append(text.lower())
                                tags_line.append(tag)
                                tags_all.append(tag)

                    texts_selected.append(texts_line)
                    tags_selected.append(tags_line)
    except FileNotFoundError as error:
        msg = "Sorry, the file" + data_file + "does not exist."
        print(msg)
        print("error:" + error)

    return texts_selected, tags_selected, tags_all


def tag_ids_map(tags_all, tags2ids_name, ids2tags_name):
    tags = list(set(tags_all))
    tags.sort()
    unique_tags = len(tags)
    ids = [i for i in range(unique_tags)]

    tags2ids = dict(zip(tags, ids))
    ids2tags = dict(zip(ids, tags))

    with open(tags2ids_name, "w") as filename:
        json.dump(tags2ids, filename)

    with open(ids2tags_name, "w") as filename:
        json.dump(ids2tags, filename)

    return tags2ids, ids2tags


def add_tagids(tags_selected, tags2ids, ids2tags):
    tagids_selected = list()
    for tags_line in tags_selected:
        tagids_line = list()
        for tag in tags_line:
            tagids_line.append(tags2ids[tag])
        tagids_selected.append(tagids_line)
    # print(tagids_selected)
    return tagids_selected


def add_text_tagid(tags_selected, tags2ids, ids2tags):
    tags_chunk = list()
    tagids_chunk = list()
    for tags_line in tags_selected:
        tag_line_chunk = list()
        tagid_line_chunk = list()
        tag_line_count = Counter(tags_line)
        if len(tag_line_count) == 1:
            tag_line_chunk.append(max(tag_line_count))
            tagid_line_chunk.append(tags2ids[max(tag_line_count)])
        else:
            del tag_line_count["O"]
            tag_line_chunk.append(max(tag_line_count))
            tagid_line_chunk.append(tags2ids[max(tag_line_count)])

        tags_chunk.append(tag_line_chunk)
        tagids_chunk.append(tagid_line_chunk)

    return tags_chunk, tagids_chunk

def save_json(json_filename, texts_selected, tags_selected, tagids_selected, tags_chunk, tagids_chunk):
    total_length = len(texts_selected)
    save_datalist = list()
    total_length = 32
    for index in range(total_length):
        item_dict = dict()
        item_dict["text"] = texts_selected[index]
        item_dict["word_tag"] = tags_selected[index]
        item_dict["word_tag_id"] = tagids_selected[index]
        item_dict["text_tag"] = tags_chunk[index]
        item_dict["text_tag_id"] = tagids_chunk[index]
        save_datalist.append(item_dict)

    with open(json_filename, 'w') as file:
        json.dump(save_datalist, file)

    return

def main(data_filename, json_filename, tags2ids_name, ids2tags_name):
    texts_selected, tags_selected, tags_all = load_texttag_file(data_filename)
    tags2ids, ids2tags = tag_ids_map(tags_all, tags2ids_name, ids2tags_name)

    tagids_selected = add_tagids(tags_selected, tags2ids, ids2tags)
    tags_chunk, tagids_chunk = add_text_tagid(tags_selected, tags2ids, ids2tags)

    save_json(json_filename, texts_selected, tags_selected, tagids_selected, tags_chunk, tagids_chunk)


if __name__ == "__main__":
    test_raw = "../data/raw_EDT/Event_detection/dev.txt"
    test_save = '../data/raw_EDT/Event_detection/dev.json'
    tags2ids_name = "../data/raw_EDT/Event_detection/tags2ids.json"
    ids2tags_name = "../data/raw_EDT/Event_detection/ids2tags.json"
    main(test_raw, test_save, tags2ids_name, ids2tags_name)