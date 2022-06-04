import torch
from torch import nn, Tensor
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import pytorch_lightning as pl

class HandsClassifier(pl.LightningModule):
    def __init__(
        self,
        hand_hidden_size: int=256,
        gru_hidden_size: int=36,
        num_layers: int=1,
        dropout: float=0.0,
        bidirectional: bool=True,
        lr: float=1e-3,
        weight_decay: float=0.0
    ):
        super().__init__()
        # self.hand_fc = nn.Sequential(
        #     nn.Linear(4*52, hand_hidden_size),
        #     nn.ELU(),
        #     # nn.Dropout(dropout),
        #     # nn.Linear(hand_hidden_size, hand_hidden_size),
        #     # nn.ELU(),
        #     # nn.Dropout(dropout),
        #     # nn.Linear(hand_hidden_size, hand_hidden_size)
        # )
        self.bidirectional = bidirectional
        self.gru = nn.GRU(
            input_size=36,
            hidden_size=gru_hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=self.bidirectional,
            batch_first=True
        )
        fc_hidden_size = hand_hidden_size + (gru_hidden_size * 2 if bidirectional else gru_hidden_size)
        # fc_hidden_size = gru_hidden_size * 2 if bidirectional else gru_hidden_size
        # fc_hidden_size = hand_hidden_size
        self.fc = nn.Sequential(
            # nn.Dropout(dropout),
            # nn.Linear(fc_hidden_size, fc_hidden_size),
            # nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden_size, 4*52),
            nn.Sigmoid()
        )

        self.lr = lr
        self.weight_decay = weight_decay

        # Treat the task as the multi-label classification problem
        self.loss = nn.BCELoss()
        self.save_hyperparameters()

    def forward(self, masked_hand, bidding, length) -> Tensor:
        # hand_features = self.hand_fc(masked_hand)
        hand_features = masked_hand

        packed_features = pack_padded_sequence(
            input=bidding,
            lengths=length.cpu(),
            batch_first=True,
            enforce_sorted=False
        )
        packed_output_features, _ = self.gru(packed_features)
        output_features, _ = pad_packed_sequence(
            sequence=packed_output_features,
            batch_first=True
        )

        if self.bidirectional:
            forward_features, backward_features = torch.chunk(output_features, 2, dim=2)
            # last_features = torch.cat((forward_features[:, -1, :], backward_features[:, 0, :]), dim=1)
            last_forward_features = torch.vstack([
                feature[sen_len-1, :]
                for feature, sen_len in zip(forward_features, length)
            ])
            last_features = torch.cat((last_forward_features, backward_features[:, 0, :]), dim=1)
        else:
            # last_features = output_features[:, -1, :]
            last_features = torch.vstack([
                feature[sen_len-1, :]
                for feature, sen_len in zip(output_features, length)
            ])

        concat_features = torch.concat([hand_features, last_features], dim=1)
        # concat_features = last_features
        # concat_features = hand_features

        return self.fc(concat_features)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def greedy_generate(self, output, hints=None):
        if hints is None:
            result = torch.zeros(output.shape)
        else:
            result = hints.clone()
            result[result < 0] = 0
            hand_cnts = torch.stack(result.split(52, dim=1)).sum(dim=2)
            total_cnts = result.sum(dim=1)
            cards_selected = torch.stack(result.split(52, dim=1)).sum(dim=0) > 0
            # Reduce the probability of selected cards and players with full hands
            output = output - cards_selected.float().repeat(1, 4) - (hand_cnts.T == 13).float().repeat_interleave(52, dim=1)

        prob_ranks = output.argsort(dim=1, descending=True)
        side_indices = torch.div(prob_ranks, 52, rounding_mode='floor')
        card_indices = prob_ranks % 52

        for i, (ranks, sides, cards) in enumerate(zip(prob_ranks, side_indices, card_indices)):
            if hints is None:
                hand_cnt = [0] * 4
                total_cnt = 0
                card_selected = [False] * 52
            else:
                hand_cnt = hand_cnts[:, i]
                total_cnt = total_cnts[i]
                card_selected = cards_selected[i, :]
            for idx, side, card in zip(ranks, sides, cards):
                if (hand_cnt[side] == 13) or card_selected[card]:
                    continue
                result[i, idx] = 1
                hand_cnt[side] += 1
                total_cnt += 1
                card_selected[card] = True
                if total_cnt == 52:
                    break
        return result

    @torch.no_grad()
    def get_accuracy(self, prediction, target, hints=None):
        both_selected = (prediction == 1) & (target == 1)
        match = prediction == target
        if hints is None:
            card_acc = (both_selected.sum(dim=1).float().mean() / 52).item()
            hand_acc = torch.stack(match.split(52, dim=1)).all(dim=2).float().mean().item()
        else:
            zero_hints = hints.clone()
            zero_hints[zero_hints < 0] = 0
            hand_both_selected = torch.stack(both_selected.split(52, dim=1))
            hand_hints = (torch.stack(zero_hints.split(52, dim=1)).sum(dim=2) == 13)
            card_acc = (hand_both_selected[~hand_hints].sum(dim=1).float().mean() / 13).item()
            hand_acc = torch.stack(match.split(52, dim=1)).all(dim=2)[~(torch.stack(zero_hints.split(52, dim=1)).sum(dim=2) == 13)].float().mean().item()
        example_acc = match.all(dim=1).float().mean().item()
        return card_acc, hand_acc, example_acc

    def training_step(self, batch, batch_idx):
        # masked_hand, self.biddings[idx], self.lengths[idx], self.hands[idx]
        masked_hand, bidding, length, target = batch
        output = self(masked_hand, bidding, length)
        loss = self.loss(input=output, target=target)
        # pred = self.greedy_generate(output, hints=masked_hand).to(target.device)
        # card_acc, hand_acc, example_acc = self.get_accuracy(pred, target, hints=masked_hand)
        self.log("train_loss", loss.item(), on_step=False, on_epoch=True, prog_bar=True, logger=True)
        # self.log("train_card_acc", card_acc, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        # self.log("train_hand_acc", hand_acc, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        # self.log("train_example_acc", example_acc, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        masked_hand, bidding, length, target = batch
        output = self(masked_hand, bidding, length)
        loss = self.loss(input=output, target=target)
        pred = self.greedy_generate(output.cpu(), hints=masked_hand.cpu())
        card_acc, hand_acc, example_acc = self.get_accuracy(pred, target.cpu(), hints=masked_hand.cpu())
        self.log("valid_loss", loss.item(), on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("valid_card_acc", card_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("valid_hand_acc", hand_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("valid_example_acc", example_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def predict_step(self, batch, batch_idx):
        masked_hand, bidding, length = batch
        output = self(masked_hand, bidding, length)
        return output
