from collections import defaultdict
from typing import Dict, List, Tuple, Union
from itertools import product
import numpy as np
from evaluate import load
from pandas.core.interchange.dataframe_protocol import DataFrame
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, \
    cohen_kappa_score, classification_report
from sympy import symbols, simplify_logic
from sympy.logic.boolalg import truth_table
from sympy.parsing.sympy_parser import parse_expr, TokenError
import re
from schema import CriterionWDecomposedTrait, TraitDecomposed
import pandas as pd


def convert_annotation(annotations: DataFrame) -> List[CriterionWDecomposedTrait]:
    trials = defaultdict(list)
    annotations = annotations.dropna(axis=0, how='all')
    for i, row in annotations.iterrows():
        row = row.to_dict()
        if pd.notna(row['Original Content']):
            if 'criterion' in locals().keys():
                if criterion['traits'][0]['trait'] == '':
                    criterion['computable'] = "Not at all"
                elif sum(trait['computable'] for trait in criterion['traits'])==len(criterion['traits']):
                    criterion['computable'] = "Completely"
                else:
                    criterion['computable'] = "Partially"
                trials[current_trial].append(criterion)
                current_trial = row['ID']
            else:
                current_trial = row['ID']
                trials[current_trial] = []
            if pd.isna(row['Constraint detail']):
                constraint_detail = None
                constraint_type = None
            elif "||" in str(row['Constraint detail']):
                constraint_detail = row['Constraint detail'].split("||")
                constraint_detail = [i.strip() for i in constraint_detail]
                constraint_type = row['Constraint type'].split("||")
                constraint_type = [i.capitalize().strip() for i in constraint_type]
            else:
                constraint_detail = [str(row['Constraint detail']).strip()]
                constraint_type = [row['Constraint type'].capitalize().strip()]
            criterion = {
                'index': len(trials[current_trial])+1,
                'in_or_ex': 'IN' if row['Inculsion/Exclusion'] == 1 else 'EX',
                'original_criterion': row['Original Content'],
                'logic_relation': '1' if pd.isna(row['Logic relation']) else str(row['Logic relation']),
                'traits': [
                    {
                        'trait_index': 1 if pd.isna(row['Decomposed number']) else row['Decomposed number'],
                        'trait': '' if pd.isna(row['Decomposed content']) else row['Decomposed content'],
                        'computable': True if pd.isna(row['Uncomputable']) else False,
                        'negation': True if pd.notna(row['Negation']) else False,
                        'rephrased_trait': '',
                        'modifier': None if pd.isna(row['Modifier']) else row['Modifier'].strip(),
                        'main_entity_content': '' if pd.isna(row['Main entity content']) else row['Main entity content'].strip(),
                        'main_entity_type': 'Other' if pd.isna(row['Main entity type']) else row['Main entity type'].capitalize().strip(),
                        'constraint_detail': constraint_detail,
                        'constraint_type': constraint_type,
                    }
                ]
            }
        else:
            if pd.isna(row['Constraint detail']):
                constraint_detail = None
                constraint_type = None
            elif "||" in str(row['Constraint detail']):
                constraint_detail = row['Constraint detail'].split("||")
                constraint_detail = [i.strip() for i in constraint_detail]
                constraint_type = row['Constraint type'].split("||")
                constraint_type = [i.capitalize().strip() for i in constraint_type]
            else:
                constraint_detail = [str(row['Constraint detail']).strip()]
                constraint_type = [row['Constraint type'].capitalize()]
            trait = {
                        'trait_index': row['Decomposed number'],
                        'trait': row['Decomposed content'],
                        'computable': True if pd.isna(row['Uncomputable']) else False,
                        'negation': True if pd.notna(row['Negation']) else False,
                        'rephrased_trait': '',
                        'modifier': None if pd.isna(row['Modifier']) else row['Modifier'].strip(),
                        'main_entity_content': '' if pd.isna(row['Main entity content']) else row['Main entity content'].strip(),
                        'main_entity_type': 'Other' if pd.isna(row['Main entity type']) else row['Main entity type'].capitalize().strip(),
                        'constraint_detail': constraint_detail,
                        'constraint_type': constraint_type,
                    }
            criterion['traits'].append(trait)
    if criterion['traits'][0]['trait'] == '':
        criterion['computable'] = "Not at all"
    elif sum(trait['computable'] for trait in criterion['traits'])==len(criterion['traits']):
        criterion['computable'] = "Completely"
    else:
        criterion['computable'] = "Partially"
    trials[current_trial].append(criterion)
    for key in trials.keys():
        trials[key] = [CriterionWDecomposedTrait(**c) for c in trials[key]]
    return trials


class Grader:
    def __init__(self, annotation: List[CriterionWDecomposedTrait], prediction: List[CriterionWDecomposedTrait], threshold=0.1,
                 model_type="microsoft/deberta-xlarge-mnli", num_layers=40, verbose=False):
        self.errors = defaultdict(list)
        self.annotation = {crit.original_criterion: crit for crit in annotation}
        self.prediction = {crit.original_criterion: crit for crit in prediction}
        self.threshold = threshold
        self.bertscore = load("bertscore")
        self.model_type = model_type
        self.num_layers = num_layers
        self.verbose = verbose

    @staticmethod
    def normalize_text(text: str) -> str:
        """Lowercase, remove extra whitespace, and replace synonyms in the text."""
        if not text:
            return ""

        # Lowercase and remove extra whitespace
        normalized_text = text.lower().strip()

        # Synonym dictionary for normalization
        synonym_dict = {
            "≤": "<=",
            "≥": ">=",
            "< ": "<",
            "= ": "=",
            "> ": ">",
            ">= ": ">=",
            "<= ": "<=",
        }

        # Replace synonyms using a dictionary
        for synonym, replacement in synonym_dict.items():
            normalized_text = re.sub(synonym, replacement, normalized_text)

        return normalized_text

    def create_identifier(self, trait):
        """Normalize the key identifiers of a trait for comparison."""
        if trait.constraint_detail is None:
            return f"{self.normalize_text(trait.main_entity_content)} | {self.normalize_text(trait.modifier)} | {self.normalize_text(trait.trait)}"
        else:
            return f"{self.normalize_text(trait.main_entity_content)} | {' || '.join([self.normalize_text(c) for c in trait.constraint_detail])} | {self.normalize_text(trait.modifier)} | {self.normalize_text(trait.trait)}"

    def bertscore_similarity(self, str1: List[str], str2: List[str]) -> List[float]:
        """Calculate Jaccard similarity between two strings."""
        str1 = [self.normalize_text(s1) for s1 in str1]
        str2 = [self.normalize_text(s2) for s2 in str2]
        results = self.bertscore.compute(predictions=str2, references=str1,
                                         model_type=self.model_type, num_layers=self.num_layers)
        return results

    @staticmethod
    def fix_unbalanced_parentheses(expression: str) -> str:
        """
        Detect and fix unbalanced parentheses in a logical expression.
        Adds missing opening or closing parentheses if necessary.
        """
        # Remove extra spaces for consistent processing
        expression = re.sub(r'\s+', ' ', expression).strip()

        # Count the number of opening and closing parentheses
        open_parens = expression.count('(')
        close_parens = expression.count(')')

        # Add missing opening parentheses if more closing ones exist
        if close_parens > open_parens:
            # Find how many we need to add at the beginning
            missing_open_parens = close_parens - open_parens
            expression = '(' * missing_open_parens + expression

        # Add missing closing parentheses if more opening ones exist
        elif open_parens > close_parens:
            # Find how many we need to add at the end
            missing_close_parens = open_parens - close_parens
            expression = expression + ')' * missing_close_parens

        return expression

    @staticmethod
    def remove_variable_in_logic(expression, variable_to_remove):
        # Remove extra spaces for consistent processing
        expression = re.sub(r'\s+', ' ', expression).strip()

        # Remove variable and operator inside parentheses
        pattern_inside_paren = rf'\(\s*\b{variable_to_remove}\b\s*(and|or)\s*(.*?)\s*\)'
        expression = re.sub(pattern_inside_paren, r'(\2)', expression)
        pattern_inside_paren_rev = rf'\(\s*(.*?)\s*(and|or)\s*\b{variable_to_remove}\b\s*\)'
        expression = re.sub(pattern_inside_paren_rev, r'(\1)', expression)

        # Remove variable between two operators
        pattern_between_ops = rf'(\band\b|\bor\b)\s+\b{variable_to_remove}\b\s+(\band\b|\bor\b)'
        expression = re.sub(pattern_between_ops, r'\1 \2', expression)

        # Remove variable at the start
        pattern_start = rf'^\b{variable_to_remove}\b\s+(\band\b|\bor\b)\s+'
        expression = re.sub(pattern_start, '', expression)

        # Remove variable at the end
        pattern_end = rf'\s+(\band\b|\bor\b)\s+\b{variable_to_remove}\b$'
        expression = re.sub(pattern_end, '', expression)

        # Remove variable with preceding operator
        pattern_var_with_prev_op = rf'(\band\b|\bor\b)\s+\b{variable_to_remove}\b'
        expression = re.sub(pattern_var_with_prev_op, '', expression)

        # Remove variable with following operator
        pattern_var_with_next_op = rf'\b{variable_to_remove}\b\s+(\band\b|\bor\b)'
        expression = re.sub(pattern_var_with_next_op, '', expression)

        # Remove variable only
        pattern_var_only = rf'\b{variable_to_remove}\b'
        expression = re.sub(pattern_var_only, '', expression)

        # Clean up redundant spaces and operators
        expression = re.sub(r'\s+', ' ', expression).strip()
        expression = re.sub(r'^(and|or)\s+', '', expression)
        expression = re.sub(r'\s+(and|or)$', '', expression)
        expression = re.sub(r'\b(and|or)\b\s+\b(and|or)\b', r'\1', expression)

        # Remove empty parentheses
        expression = re.sub(r'\(\s*\)', '', expression)

        # Simplify parentheses around single variables
        expression = re.sub(r'\(\s*(\w+)\s*\)', r'\1', expression)

        return expression

    def clear_results_for_not_computable(self, criterion):
        for trait in criterion.traits:
            if not trait.computable:
                trait.modifier = None
                trait.main_entity_content = ''
                trait.main_entity_type = 'Other'
                trait.constraint_detail = None
                trait.constraint_type = None
                if str(trait.trait_index) in criterion.logic_relation:
                    criterion.logic_relation = self.remove_variable_in_logic(criterion.logic_relation.lower(),
                                                                             str(trait.trait_index))
            if trait.negation:
                pattern = rf'\b{trait.trait_index}\b'
                criterion.logic_relation = re.sub(pattern, rf'not {trait.trait_index}', criterion.logic_relation)
        return criterion

    def calculate_constraint_metrics(self, ann_trait: TraitDecomposed, pred_trait: TraitDecomposed) -> Dict[str, Union[str, int]]:
        """Match traits from annotation and prediction using exact or fuzzy matching."""
        matched_pairs = []
        if pred_trait.constraint_detail is None and ann_trait.constraint_detail is None:
            return {'fp': 0, 'fn': 0, 'ann': [], 'pred': [], 'ann_type': ["None"], 'pred_type': ["None"]}
        elif pred_trait.constraint_detail is None:
            if self.verbose:
                print("Anno constraint not matched:", ann_trait.constraint_detail)
            return {'fp': 0, 'fn': len(ann_trait.constraint_detail), 'ann': [], 'pred': [], 'ann_type': [],
                    'pred_type': []}
        elif ann_trait.constraint_detail is None:
            if self.verbose:
                print("Pred constraint not matched:", pred_trait.constraint_detail)
            return {'fp': len(pred_trait.constraint_detail), 'fn': 0, 'ann': [], 'pred': [], 'ann_type': [],
                    'pred_type': []}
        else:
            unmatched_preds = [(detail, typ) for detail, typ in
                               zip(pred_trait.constraint_detail, pred_trait.constraint_type)]
            unmatched_annos = [(detail, typ) for detail, typ in
                               zip(ann_trait.constraint_detail, ann_trait.constraint_type)]

            # Perform exact matching first
            for ann_trait in unmatched_annos.copy():
                for pred_trait in unmatched_preds.copy():
                    if ann_trait == pred_trait:
                        matched_pairs.append((ann_trait, pred_trait))
                        unmatched_preds.remove(pred_trait)
                        unmatched_annos.remove(ann_trait)
                        break

            similarities = []
            # Perform fuzzy matching for unmatched annotation traits
            for ann_trait in unmatched_annos.copy():
                if not any(ann_trait == pair[0] for pair in matched_pairs):
                    # Collect similarities for all unmatched prediction traits
                    for pred_trait in unmatched_preds.copy():
                        similarity = self.bertscore_similarity([' '.join(ann_trait)], [' '.join(pred_trait)])['f1'][0]
                        similarities.append(((ann_trait, pred_trait), similarity))

                    # Sort predicted traits by similarity in descending order
            similarities.sort(key=lambda x: x[1], reverse=True)

            # Try to match the best one if similarity exceeds the threshold
            for best_match, best_similarity in similarities:
                if best_similarity >= self.threshold and best_match[0] in unmatched_annos and best_match[
            1] in unmatched_preds:
                    matched_pairs.append(best_match)
                    unmatched_preds.remove(best_match[1])
                    unmatched_annos.remove(best_match[0])

            if self.verbose:
                print("Matched constraints")
                for pair in matched_pairs:
                    print("anno:", pair[0], "&pred:", pair[1])
                if len(unmatched_annos) > 0 or len(unmatched_preds) > 0:
                    print("Anno constraint not matched", unmatched_annos)
                    print("Pred constraint not matched", unmatched_preds)

            return {'fp': len(unmatched_preds),
                    'fn': len(unmatched_annos),
                    'ann': [pair[0][0] for pair in matched_pairs],
                    'pred': [pair[1][0] for pair in matched_pairs],
                    'ann_type': [pair[0][1] for pair in matched_pairs],
                    'pred_type': [pair[1][1] for pair in matched_pairs]}

    def calculate_all_metrics(self) -> Dict[str, Dict[str, float]]:
        # Initialize metrics
        trait_metrics = defaultdict(float)
        logic_relation_metrics = defaultdict(float)
        values = defaultdict(list)

        # Track counts
        total_prediction = len(self.prediction)
        logic_relation_exact_match = 0

        trait_metrics['computable_tp'] = 0
        trait_metrics['computable_fp'] = 0
        trait_metrics['computable_fn'] = 0
        trait_metrics['computable_tn'] = 0

        unmatched_pred_criteria = set(self.prediction.keys()) - set(self.annotation.keys())
        unmatched_anno_criteria = set(self.annotation.keys()) - set(self.prediction.keys())

        comb = product(unmatched_pred_criteria, unmatched_anno_criteria)
        comb = list(comb)
        pred_criteria_list, anno_criteria_list = zip(*comb)
        pred_criteria_list = list(pred_criteria_list)
        anno_criteria_list = list(anno_criteria_list)
        score = self.bertscore_similarity(anno_criteria_list, pred_criteria_list)

        for original_criterion, ann_criterion in self.annotation.items():
            if original_criterion not in self.prediction:
                if self.verbose:
                    print("prediction not exact found:", original_criterion)
                idx = [i for i, anno_criterion in enumerate(anno_criteria_list) if anno_criterion == original_criterion]
                f1 = np.array(score['f1'])[idx]
                idx_max = f1.argmax()
                tmp_max = f1.max()
                if tmp_max>0.9:
                    original_criterion = pred_criteria_list[idx[idx_max]]
                else:
                    continue  # Skip if there is no prediction for this criterion
            if self.verbose:
                print("\n\n")
            pred_criterion = self.prediction[original_criterion]

            if pred_criterion.logic_relation is None:
                pred_criterion.logic_relation = '1'

            # 1. Trait Matching
            matched_pairs, anno_unmatched, pred_unmatched = self.match_traits(ann_criterion.traits,
                                                                              pred_criterion.traits)
            if len(anno_unmatched) > 0 or len(pred_unmatched) > 0:
                self.errors['unmatched'].append((anno_unmatched, pred_unmatched))
            trait_metrics['trait_count'] += len(ann_criterion.traits)
            trait_metrics['pred_trait_count'] += len(pred_criterion.traits)
            trait_metrics['trait_tp'] += len(matched_pairs)
            trait_metrics['trait_fp'] += len(pred_criterion.traits) - len(matched_pairs)
            trait_metrics['trait_fn'] += len(ann_criterion.traits) - len(matched_pairs)

            trait_metrics['computable_fp'] += len([trait for trait in pred_unmatched if trait.computable])
            trait_metrics['computable_fn'] += len([trait for trait in anno_unmatched if trait.computable])

            # Update logic relation with matched trait indices
            matched_preds = [pair[1] for pair in matched_pairs]
            unmatched_pred = [trait for trait in pred_criterion.traits if trait not in matched_preds]

            pred_criterion.logic_relation = re.sub(r'(\d+)', 'x' + r'\1', pred_criterion.logic_relation)

            for i, pred_trait in enumerate(unmatched_pred):
                pred_criterion.logic_relation = re.sub(r'\b' + 'x' + str(pred_trait.trait_index) + r'\b', str(999 + i),
                                                       pred_criterion.logic_relation)

            values['matched_pairs'] += matched_pairs
            # Evaluate each field of the matched traits
            for ann_trait, pred_trait in matched_pairs:
                pred_criterion.logic_relation = re.sub(r'\b' + 'x' + str(pred_trait.trait_index) + r'\b',
                                                       str(ann_trait.trait_index), pred_criterion.logic_relation)
                for field in ["modifier", "main_entity_content", "main_entity_type"]:
                    if getattr(ann_trait, field) is None:
                        values[field + "_ann"].append("")
                    else:
                        values[field + "_ann"].append(getattr(ann_trait, field).lower())
                    if getattr(pred_trait, field) is None:
                        values[field + "_pred"].append("")
                    else:
                        values[field + "_pred"].append(getattr(pred_trait, field).lower())
                if ann_trait.computable:
                    constraint_metrics = self.calculate_constraint_metrics(ann_trait, pred_trait)
                    trait_metrics['trait_fp'] += constraint_metrics['fp']
                    trait_metrics['trait_fn'] += constraint_metrics['fn']
                    values["constraint_detail_ann"] += constraint_metrics['ann']
                    values["constraint_detail_pred"] += constraint_metrics['pred']
                    values["constraint_type_ann"] += constraint_metrics['ann_type']
                    values["constraint_type_pred"] += constraint_metrics['pred_type']

                for field in ["computable", "negation"]:
                    values[field + "_ann"].append(getattr(ann_trait, field))
                    values[field + "_pred"].append(getattr(pred_trait, field))
            # 2. Logic Relation Matching
            if ann_criterion.logic_relation and pred_criterion.logic_relation:
                if self.verbose:
                    print("Annotation logic relation", ann_criterion.logic_relation)
                    print("Prediction logic relation", pred_criterion.logic_relation)
                em, tn, fp, fn, tp = self._compare_logic_relations(ann_criterion.logic_relation,
                                                                   pred_criterion.logic_relation)
                logic_relation_exact_match += int(em)
                logic_relation_metrics['tn'] += tn
                logic_relation_metrics['fp'] += fp
                logic_relation_metrics['fn'] += fn
                logic_relation_metrics['tp'] += tp
            elif ann_criterion.logic_relation is None and pred_criterion.logic_relation is None:
                logic_relation_exact_match += 1
                logic_relation_metrics['tn'] += 1
            elif ann_criterion.logic_relation:
                logic_relation_metrics['fn'] += 1
            elif pred_criterion.logic_relation:
                logic_relation_metrics['fp'] += 1

        # Calculate trait-level metrics
        for field in ["computable", "negation"]:
            if self.verbose:
                print(field, confusion_matrix(values[field + '_ann'], values[field + '_pred']))
            err = (values[field + '_ann'] != values[field + '_pred'])
            self.errors[field].append(values['matched_pairs'][err])
            trait_metrics[field + '_precision'] = precision_score(values[field + '_ann'], values[field + '_pred'],
                                                                  average='weighted')
            trait_metrics[field + '_recall'] = recall_score(values[field + '_ann'], values[field + '_pred'],
                                                            average='weighted')
            trait_metrics[field + '_f1_score'] = f1_score(values[field + '_ann'], values[field + '_pred'],
                                                          average='weighted')
            trait_metrics[field + '_exact_match_rate'] = accuracy_score(values[field + '_ann'], values[field + '_pred'])
            trait_metrics[field + '_kappa'] = cohen_kappa_score(values[field + '_ann'], values[field + '_pred'])

        for field in ["modifier", "main_entity_content"]:
            none_mask = ~(np.array((pd.Series(values[field + '_ann'])=="")&(pd.Series(values[field + '_pred'])=="")))
            computable_mask = np.array(values['computable_ann'])

            prediction = np.array(values[field + '_pred'])[none_mask&computable_mask]
            annotation = np.array(values[field + '_ann'])[none_mask&computable_mask]
            matched_pair = np.array(values['matched_pairs'])[none_mask&computable_mask]
            results = self.bertscore_similarity(prediction,annotation)
            err = np.array(results['f1']) < 0.75
            self.errors[field].append(matched_pair[err])
            trait_metrics[field + '_f1_score'] = np.mean(results['f1'])
            trait_metrics[field + '_precision'] = np.mean(results['precision'])
            trait_metrics[field + '_recall'] = np.mean(results['recall'])
            trait_metrics[field + '_exact_match_rate'] = accuracy_score(values[field + '_ann'], values[field + '_pred'])
        results = self.bertscore_similarity(values['constraint_detail_pred'], values['constraint_detail_ann'])
        err = np.array(results['f1']) < 0.75
        self.errors['constraint'].append(np.array(list(zip(*[values['constraint_detail_ann'], values['constraint_detail_pred']])))[err])
        trait_metrics['constraint_detail_f1_score'] = np.mean(results['f1'])
        trait_metrics['constraint_detail_precision'] = np.mean(results['precision'])
        trait_metrics['constraint_detail_recall'] = np.mean(results['recall'])
        trait_metrics['constraint_detail_exact_match_rate'] = accuracy_score(values['constraint_detail_ann'],
                                                                                values['constraint_detail_pred'])

        # Calculate criterion-level metrics
        trait_metrics['precision'] = trait_metrics['trait_tp'] / (
                trait_metrics['trait_tp'] + trait_metrics['trait_fp']) if trait_metrics[
                                                                              'trait_tp'] + \
                                                                          trait_metrics[
                                                                              'trait_fp'] > 0 else 0
        trait_metrics['recall'] = trait_metrics['trait_tp'] / (trait_metrics['trait_tp'] + trait_metrics['trait_fn']) if \
            trait_metrics[
                'trait_tp'] + \
            trait_metrics[
                'trait_fn'] > 0 else 0
        trait_metrics['f1_score'] = 2 * (trait_metrics['precision'] * trait_metrics['recall']) / (
                trait_metrics['precision'] + trait_metrics['recall']) if trait_metrics['precision'] + trait_metrics[
            'recall'] > 0 else 0
        trait_metrics['exact_match_rate'] = trait_metrics['trait_tp'] / trait_metrics['trait_count'] if trait_metrics[
                                                                                                            'trait_count'] > 0 else 0

        # Calculate logic relation metrics
        logic_relation_metrics[
            'exact_match_rate'] = logic_relation_exact_match / total_prediction if total_prediction > 0 else 0
        logic_relation_metrics['precision'] = logic_relation_metrics['tp'] / (
                logic_relation_metrics['tp'] + logic_relation_metrics['fp']) if logic_relation_metrics['tp'] + \
                                                                                logic_relation_metrics[
                                                                                    'fp'] > 0 else 0
        logic_relation_metrics['recall'] = logic_relation_metrics['tp'] / (
                logic_relation_metrics['tp'] + logic_relation_metrics['fn']) if logic_relation_metrics['tp'] + \
                                                                                logic_relation_metrics[
                                                                                    'fn'] > 0 else 0
        logic_relation_metrics['f1_score'] = 2 * (
                logic_relation_metrics['precision'] * logic_relation_metrics['recall']) / (
                                                     logic_relation_metrics['precision'] + logic_relation_metrics[
                                                 'recall']) if logic_relation_metrics['precision'] + \
                                                               logic_relation_metrics['recall'] > 0 else 0

        # Return combined report
        return {
            "trait_metrics": trait_metrics,
            "logic_relation_metrics": logic_relation_metrics,
            "values": values
        }

    def match_traits(self, ann_traits: List[TraitDecomposed], pred_traits: List[TraitDecomposed]) -> \
            Tuple[List[Tuple[TraitDecomposed, TraitDecomposed]], List[str], List[str]]:
        """Match traits from annotation and prediction using exact or fuzzy matching."""
        matched_pairs = []
        unmatched_preds = pred_traits.copy()
        unmatched_annos = ann_traits.copy()
        identifier_cache = {}

        # Perform exact matching first
        for ann_trait in unmatched_annos.copy():
            if f"ann_{ann_trait.trait_index}" not in identifier_cache:
                identifier_cache[f"ann_{ann_trait.trait_index}"] = self.create_identifier(ann_trait)
            ann_id = identifier_cache[f"ann_{ann_trait.trait_index}"]

            for pred_trait in unmatched_preds.copy():
                if f"pred_{pred_trait.trait_index}" not in identifier_cache:
                    identifier_cache[f"pred_{pred_trait.trait_index}"] = self.create_identifier(pred_trait)
                pred_id = identifier_cache[f"pred_{pred_trait.trait_index}"]

                if ann_id == pred_id:
                    matched_pairs.append((ann_trait, pred_trait))
                    unmatched_preds.remove(pred_trait)
                    unmatched_annos.remove(ann_trait)
                    break

        # Perform fuzzy matching for unmatched annotation traits
        similarities = []
        for ann_trait in unmatched_annos.copy():
            if not any(ann_trait == pair[0] for pair in matched_pairs):
                if f"ann_{ann_trait.trait_index}" not in identifier_cache:
                    identifier_cache[f"ann_{ann_trait.trait_index}"] = self.create_identifier(ann_trait)
                ann_id = identifier_cache[f"ann_{ann_trait.trait_index}"]

                for pred_trait in unmatched_preds.copy():
                    pred_id = identifier_cache[f"pred_{pred_trait.trait_index}"]
                    similarity = self.bertscore_similarity([ann_id], [pred_id])['f1'][0]
                    similarities.append(((ann_trait, pred_trait), similarity))

        # Sort predicted traits by similarity in descending order
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Try to match the best one if similarity exceeds the threshold
        for best_match, best_similarity in similarities:
            if best_similarity >= self.threshold and best_match[0] in unmatched_annos and best_match[
                1] in unmatched_preds:
                matched_pairs.append(best_match)
                unmatched_preds.remove(best_match[1])
                unmatched_annos.remove(best_match[0])

        if self.verbose:
            print("Matched pairs")
            for pair in matched_pairs:
                print("anno:", pair[0].trait, "&pred:", pair[1].trait)
        anno_unmatched = [trait for trait in ann_traits if trait not in [pair[0] for pair in matched_pairs]]
        pred_unmatched = [trait for trait in pred_traits if trait not in [pair[1] for pair in matched_pairs]]
        if self.verbose and (len(anno_unmatched) > 0 or len(pred_unmatched) > 0):
            print("Anno trait not matched",
                  [trait.trait for trait in ann_traits if trait not in [pair[0] for pair in matched_pairs]])
            print("Pred trait not matched",
                  [trait.trait for trait in pred_traits if trait not in [pair[1] for pair in matched_pairs]])

        return matched_pairs, anno_unmatched, pred_unmatched

    def convert_to_dnf(self, expr_str: str):
        # Replace logical operators with Python bitwise operators
        expr_str = expr_str.lower()
        expr_str = expr_str.replace('[', '(')
        expr_str = expr_str.replace(']', ')')
        expr_str = expr_str.replace('{', '(')
        expr_str = expr_str.replace('}', ')')
        expr_str = expr_str.replace('and', '&')
        expr_str = expr_str.replace('or', '|')
        expr_str = expr_str.replace('not', '~')

        # Extract unique variable names using regex
        tokens = set(re.findall(r'\b\w+\b', expr_str))
        # Remove logical operators from tokens
        logical_ops = {'&', '|', '~'}
        tokens = tokens - logical_ops
        for token in tokens:
            expr_str = re.sub(rf'\b{token}\b', rf'x{token}', expr_str)
        tokens = ['x' + tokens for tokens in tokens]

        # Create symbols for each variable
        symbols_dict = {token: symbols(token) for token in tokens}
        if self.verbose:
            print("expr_str:", expr_str)
        # Parse the expression into a sympy expression
        try:
            expr = parse_expr(expr_str, local_dict=symbols_dict)
        except TokenError:
            expr_str = self.fix_unbalanced_parentheses(expr_str)
            expr = parse_expr(expr_str, local_dict=symbols_dict)

        # Convert the expression to DNF
        dnf_expr = simplify_logic(expr, form='dnf')
        if self.verbose:
            print("dnf_expr:", dnf_expr)
        return dnf_expr, symbols_dict.values()

    def _compare_logic_relations(self, ann_relation: str, pred_relation: str) -> Tuple[bool, int, int, int, int]:
        """Compare logic relations by normalizing and checking for equivalence."""
        # Normalize and compare using string comparison for now
        em = ann_relation.strip().lower() == pred_relation.strip().lower()
        if em and ann_relation == '1':
            return em, 0, 0, 0, 0
        else:
            ann_dnf, ann_trait_id = self.convert_to_dnf(ann_relation)
            try:
                pred_dnf, pred_trait_id = self.convert_to_dnf(pred_relation)

                all_trait_ids = set(ann_trait_id) | set(pred_trait_id)

                truth_table_ann = list(zip(*truth_table(ann_dnf, all_trait_ids)))[1]
                truth_table_pred = list(zip(*truth_table(pred_dnf, all_trait_ids)))[1]

                truth_table_ann = [bool(val) for val in truth_table_ann]
                truth_table_pred = [bool(val) for val in truth_table_pred]

                tn, fp, fn, tp = confusion_matrix(truth_table_ann, truth_table_pred).ravel()

                return em, tn, fp, fn, tp
            except (SyntaxError, TypeError):
                return em, 0, 0, len(ann_trait_id), 0

    def generate_report(self):
        """Generate a report based on the calculated metrics."""
        metrics = self.calculate_all_metrics()
        trait_metrics = metrics['trait_metrics']
        logic_metrics = metrics['logic_relation_metrics']
        values = metrics['values']

        report = f"""
        Evaluation Report:
        ==================
        1. Trial Extraction Evaluation:
           - Precision: {trait_metrics['precision']:.2f}
           - Recall: {trait_metrics['recall']:.2f}
           - F1 Score: {trait_metrics['f1_score']:.2f}

        2. Logic Relation Evaluation:
            - Precision: {logic_metrics['precision']:.2f}
            - Recall: {logic_metrics['recall']:.2f}
            - F1 Score: {logic_metrics['f1_score']:.2f}

        3. Trait-Level Evaluation:

        BertScore:
           - Main Entity Precision: {trait_metrics['main_entity_content_precision']:.2f}
           - Main Entity Recall: {trait_metrics['main_entity_content_recall']:.2f}
           - Main Entity F1 Score: {trait_metrics['main_entity_content_f1_score']:.2f}

           - Constraint Precision: {trait_metrics['constraint_detail_precision']:.2f}
           - Constraint Recall: {trait_metrics['constraint_detail_recall']:.2f}
           - Constraint F1 Score: {trait_metrics['constraint_detail_f1_score']:.2f}

           - Modifier Precision: {trait_metrics['modifier_precision']:.2f}
           - Modifier Recall: {trait_metrics['modifier_recall']:.2f}
           - Modifier F1 Score: {trait_metrics['modifier_f1_score']:.2f}

        Classification:
           - Main Entity Type
             {classification_report(values['main_entity_type_ann'], values['main_entity_type_pred'])}

           - Constraint Type
             {classification_report(values['constraint_type_ann'], values['constraint_type_pred'])}

           - Computable Precision: {trait_metrics['computable_precision']:.2f}
           - Computable Recall: {trait_metrics['computable_recall']:.2f}
           - Computable F1 Score: {trait_metrics['computable_f1_score']:.2f}

           - Negation Precision: {trait_metrics['negation_precision']:.2f}
           - Negation Recall: {trait_metrics['negation_recall']:.2f}
           - Negation F1 Score: {trait_metrics['negation_f1_score']:.2f}


        4. Trait Confusion Matrix:
              - TP: {trait_metrics['trait_tp']}
              - FP: {trait_metrics['trait_fp']}
              - FN: {trait_metrics['trait_fn']}

        Total Criteria Evaluated: {len(self.prediction)}
        """

        return report