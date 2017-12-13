import word2gm_loader
import sys
import numpy as np

def evaluate(model_dir, test_file_path):
    # load model
    print("loading prediction model...")
    model = word2gm_loader.Word2GM(model_dir, ckpt_file=None, verbose=True)

    # load test instances
    print("loading test instances...")
    index = dict()
    test_instances = []
    with open(test_file_path, "r") as f:
        for id, line in enumerate(f):
            line_split = line.split(";", 2)
            test_instances.append((line_split[0], line_split[1], line_split[2]))
            try:
                meaning_map = index[line_split[0]]
            except KeyError:
                meaning_map = dict()
                index[line_split[0]] = meaning_map
            try:
                meaning_map[line_split[1]] = meaning_map[line_split[1]] + [id]
            except KeyError:
                meaning_map[line_split[1]] = [id]

    # predict senses
    print("performing sense predictions..")
    unknown_acronyms = 0
    predictions = [-1] * len(test_instances)
    for i, inst in enumerate(test_instances):
        acronym = inst[0].lower()
        text = inst[2]
        try:
            acronym_id = model.word2id[acronym]
            text_ids = model.words_to_idxs(text.split(), discard_unk=True)
            try: # prevent failure when the acronym_id is not contained in text_ids (probably lemmatization issues for the acronym is used in plural (+s) in the text)
                text_ids.remove(acronym_id) # we remove the acronym from the text sample so that it doesn't influence
                                            # into determining the best mixture component itself
            except:
                pass
            predictions[i] = model.find_best_cluster(acronym_id, text_ids, criterion='max', verbose=False)
        except KeyError:
            print("acronym " + acronym + " is unknown to the model, skipping test instance.")
            unknown_acronyms += 1

    # associate mixture component number from model with class label from test data and compute precision
    print("associating predictions with labels and computing precision...")
    results = dict()
    for acronym in index:
        label_association = dict()
        precisions = dict()
        allocation_matrix = np.ones((len(index[acronym]), model.num_mixtures))
        sense_id = dict()

        # assign to each sense/label the component for which the relative amount of samples carrying that label is hights, i.e., max_{component}( #samples_component_and_label / #samples_component )
        # note[Lukas]: there are probably far better solutions than this quick and dirty approach
        for i, sense in enumerate(index[acronym]):
            sense_id[sense] = i
            for inst_id in index[acronym][sense]:
                prediction = predictions[inst_id]
                if prediction > -1:
                    allocation_matrix[i][prediction] += 1
        best_comp_per_sense = np.argmax(allocation_matrix / np.sum(allocation_matrix, axis=0), axis=1)

        # compute precision for all senses (note: no penalty for assigning same component to different senses. this should be penalized but isn't right now)
        for sense in index[acronym]:
            correct_class = best_comp_per_sense[sense_id[sense]]
            label_association[sense] = correct_class
            correct_prediction_count = sum([1 if predictions[inst_id] == correct_class else 0 for inst_id in index[acronym][sense]])
            precision = correct_prediction_count / len(index[acronym][sense])
            precisions[sense] = (correct_prediction_count, precision)
        results[acronym] = (label_association, precisions)

    return results

if __name__ == '__main__':
    model_dir = sys.argv[1]
    test_file_path = sys.argv[2]
    print(evaluate(model_dir, test_file_path))
