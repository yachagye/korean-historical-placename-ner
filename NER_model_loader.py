"""
NER 모델 로더 유틸리티
체크포인트에서 BERT + classifier를 복원하는 공통 로직
"""

import torch
import torch.nn as nn
from typing import Tuple
from transformers import AutoTokenizer, AutoModel


def load_bert_classifier_from_checkpoint(
    checkpoint_path: str,
    device: torch.device,
    num_labels: int,
) -> Tuple[AutoTokenizer, AutoModel, nn.Module, nn.Module, float, str]:
    """
    체크포인트에서 모델 구성 요소들을 로드

    Args:
        checkpoint_path: 학습된 체크포인트 경로
        device: 디바이스
        num_labels: 라벨 개수

    Returns:
        (tokenizer, bert, classifier, dropout, threshold, model_name)
    """
    # 체크포인트 로드
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # 하이퍼파라미터 추출
    hparams = checkpoint['hyper_parameters']
    model_name = hparams['model_name']
    threshold = hparams.get('threshold', 0.5)
    dropout_rate = hparams.get('dropout_rate', 0.1)

    # 토크나이저/모델 생성
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    bert = AutoModel.from_pretrained(model_name)
    dropout = nn.Dropout(dropout_rate)
    classifier = nn.Linear(bert.config.hidden_size, num_labels)

    # state_dict에서 가중치 필터링
    state_dict = checkpoint['state_dict']

    bert_state_dict = {}
    classifier_state_dict = {}

    for key, value in state_dict.items():
        if key.startswith('bert.'):
            new_key = key[5:]  # 'bert.' 제거
            bert_state_dict[new_key] = value
        elif key.startswith('classifier.'):
            new_key = key[11:]  # 'classifier.' 제거
            classifier_state_dict[new_key] = value

    # 가중치 로드
    bert.load_state_dict(bert_state_dict)
    classifier.load_state_dict(classifier_state_dict)

    # 디바이스로 이동 및 평가 모드 설정
    bert = bert.to(device)
    classifier = classifier.to(device)
    dropout = dropout.to(device)

    bert.eval()
    classifier.eval()
    dropout.eval()

    return tokenizer, bert, classifier, dropout, threshold, model_name
