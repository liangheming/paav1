import torch
from utils.boxs_utils import box_iou
from losses.commons import focal_loss, IOULoss, BoxSimilarity
import sklearn.mixture as skm

INF = 100000000


class BoxCoder(object):
    def __init__(self, weights=None):
        super(BoxCoder, self).__init__()
        if weights is None:
            weights = [0.1, 0.1, 0.2, 0.2]
        self.weights = torch.tensor(data=weights, requires_grad=False)

    def encoder(self, anchors, gt_boxes):
        """
        :param gt_boxes:[box_num, 4]
        :param anchors: [box_num, 4]
        :return:
        """
        if self.weights.device != anchors.device:
            self.weights = self.weights.to(anchors.device)
        anchors_wh = anchors[..., [2, 3]] - anchors[..., [0, 1]]
        anchors_xy = anchors[..., [0, 1]] + 0.5 * anchors_wh
        gt_wh = (gt_boxes[..., [2, 3]] - gt_boxes[..., [0, 1]]).clamp(min=1.0)
        gt_xy = gt_boxes[..., [0, 1]] + 0.5 * gt_wh
        delta_xy = (gt_xy - anchors_xy) / anchors_wh
        delta_wh = (gt_wh / anchors_wh).log()

        delta_targets = torch.cat([delta_xy, delta_wh], dim=-1) / self.weights

        return delta_targets

    def decoder(self, predicts, anchors):
        """
        :param predicts: [anchor_num, 4] or [bs, anchor_num, 4]
        :param anchors: [anchor_num, 4]
        :return: [anchor_num, 4] (x1,y1,x2,y2)
        """
        if self.weights.device != anchors.device:
            self.weights = self.weights.to(anchors.device)
        anchors_wh = anchors[:, [2, 3]] - anchors[:, [0, 1]]
        anchors_xy = anchors[:, [0, 1]] + 0.5 * anchors_wh
        scale_reg = predicts * self.weights
        scale_reg[..., :2] = anchors_xy + scale_reg[..., :2] * anchors_wh
        scale_reg[..., 2:] = scale_reg[..., 2:].exp() * anchors_wh
        scale_reg[..., :2] -= (0.5 * scale_reg[..., 2:])
        scale_reg[..., 2:] = scale_reg[..., :2] + scale_reg[..., 2:]

        return scale_reg


class Matcher(object):
    BELOW_LOW_THRESHOLD = -1
    BETWEEN_THRESHOLDS = -2

    def __init__(self, iou_thresh=0.5, ignore_iou=0.4, allow_low_quality_matches=True):
        self.iou_thresh = iou_thresh
        self.ignore_iou = ignore_iou
        self.allow_low_quality_matches = allow_low_quality_matches

    @torch.no_grad()
    def __call__(self, anchors, gt_boxes):
        ret = list()
        for idx, gt_box in enumerate(gt_boxes):
            if len(gt_box) == 0:
                continue
            ori_match = None
            gt_anchor_iou = box_iou(gt_box[..., 1:], anchors)
            match_val, match_idx = gt_anchor_iou.max(dim=0)
            if self.allow_low_quality_matches:
                ori_match = match_idx.clone()
            match_idx[match_val < self.ignore_iou] = self.BELOW_LOW_THRESHOLD
            match_idx[(match_val >= self.ignore_iou) & (match_val < self.iou_thresh)] = self.BETWEEN_THRESHOLDS
            if self.allow_low_quality_matches:
                self.set_low_quality_matches_(match_idx, ori_match, gt_anchor_iou)
            ret.append((idx, match_idx))
        return ret

    @staticmethod
    def set_low_quality_matches_(matches, ori_matches, gt_anchor_iou):
        highest_quality_foreach_gt, _ = gt_anchor_iou.max(dim=1)
        # [num,2](gt_idx,anchor_idx)
        gt_pred_pairs_of_highest_quality = torch.nonzero(
            gt_anchor_iou == highest_quality_foreach_gt[:, None], as_tuple=False
        )
        pred_inds_to_update = gt_pred_pairs_of_highest_quality[:, 1]
        matches[pred_inds_to_update] = ori_matches[pred_inds_to_update]


class PAALoss(object):
    def __init__(self,
                 top_k=9,
                 alpha=0.25,
                 gamma=2.0,
                 iou_thresh=0.1,
                 allow_low_quality_matches=True,
                 iou_type="giou",
                 iou_loss_weight=0.5,
                 reg_loss_weight=1.3):
        self.top_k = top_k
        self.alpha = alpha
        self.gamma = gamma
        self.iou_loss_weight = iou_loss_weight
        self.reg_loss_weight = reg_loss_weight
        self.matcher = Matcher(iou_thresh=iou_thresh,
                               ignore_iou=iou_thresh,
                               allow_low_quality_matches=allow_low_quality_matches)
        self.box_coder = BoxCoder()
        self.iou_loss = IOULoss(iou_type=iou_type)
        self.box_similarity = BoxSimilarity(iou_type="iou")
        self.bce = torch.nn.BCEWithLogitsLoss(reduction="sum")

    def paa_assign(self, match_idx, combine_loss, num_gt, pred_num_per_layer):
        device = combine_loss.device
        match_gt_idx = torch.full_like(match_idx, fill_value=self.matcher.BELOW_LOW_THRESHOLD)
        for gt_idx in range(num_gt):
            start_idx = 0
            candidate_anchor_ids = list()
            for pred_num in pred_num_per_layer:
                layer_match_idx = match_idx[start_idx:start_idx + pred_num]
                layer_combine_loss = combine_loss[start_idx:start_idx + pred_num]
                valid_layer_idx = ((layer_match_idx == gt_idx) & (layer_match_idx >= 0)).nonzero(
                    as_tuple=False).squeeze(-1)
                if valid_layer_idx.numel() > 0:
                    _, top_k = layer_combine_loss[valid_layer_idx].topk(
                        min(valid_layer_idx.numel(), self.top_k), largest=False
                    )
                    top_k_idx_per_layer = valid_layer_idx[top_k] + start_idx
                    candidate_anchor_ids.append(top_k_idx_per_layer)
                start_idx += pred_num
            if candidate_anchor_ids:
                candidate_anchor_ids = torch.cat(candidate_anchor_ids)

            if len(candidate_anchor_ids) and candidate_anchor_ids.numel() > 1:
                candidate_loss = combine_loss[candidate_anchor_ids]
                candidate_loss, inds = candidate_loss.sort()
                candidate_loss = candidate_loss.view(-1, 1).cpu().numpy()
                min_loss, max_loss = candidate_loss.min(), candidate_loss.max()
                means_init = [[min_loss], [max_loss]]
                weights_init = [0.5, 0.5]
                precisions_init = [[[1.0]], [[1.0]]]
                gmm = skm.GaussianMixture(2,
                                          weights_init=weights_init,
                                          means_init=means_init,
                                          precisions_init=precisions_init)
                gmm.fit(candidate_loss)
                components = gmm.predict(candidate_loss)
                scores = gmm.score_samples(candidate_loss)
                components = torch.from_numpy(components).to(device)
                scores = torch.from_numpy(scores).to(device)
                fgs = components == 0
                bgs = components == 1
                if fgs.nonzero(as_tuple=False).squeeze(-1).numel() > 0:
                    fg_max_score = scores[fgs].max().item()
                    fg_max_idx = (fgs & (scores == fg_max_score)).nonzero(as_tuple=False).squeeze(-1).min()
                    is_neg = inds[fgs | bgs]
                    is_pos = inds[:fg_max_idx + 1]
                else:
                    is_pos = inds
                    is_neg = None
                if is_neg is not None:
                    neg_idx = candidate_anchor_ids[is_neg]
                    match_gt_idx[neg_idx] = self.matcher.BELOW_LOW_THRESHOLD
                pos_idx = candidate_anchor_ids[is_pos]
                match_gt_idx[pos_idx] = gt_idx
        return match_gt_idx

    def __call__(self, cls_predicts, reg_predicts, iou_predicts, anchors, targets):
        pred_num_per_layer = [item.shape[1] for item in reg_predicts]
        cls_predicts = torch.cat([item for item in cls_predicts], dim=1)
        reg_predicts = torch.cat([item for item in reg_predicts], dim=1)
        all_anchors = torch.cat([item for item in anchors])
        iou_predicts = torch.cat([item for item in iou_predicts], dim=1)
        gt_boxes = targets['target'].split(targets['batch_len'])

        if cls_predicts.dtype == torch.float16:
            cls_predicts = cls_predicts.float()
        if iou_predicts.dtype == torch.float16:
            iou_predicts = iou_predicts.float()

        match_batch_idx = list()
        match_anchor_idx = list()
        match_gt_idx = list()

        matches = self.matcher(all_anchors, gt_boxes)
        for bid, match_idx in matches:
            cls_target = gt_boxes[bid][:, 0].long()[match_idx.clamp(min=0)]
            cls_predicts_item = cls_predicts[bid].detach()
            reg_predicts_item = reg_predicts[bid].detach()
            predicts_box = self.box_coder.decoder(predicts=reg_predicts_item[match_idx >= 0],
                                                  anchors=all_anchors[match_idx >= 0])
            cls_loss = focal_loss(cls_predicts_item.sigmoid().clamp(min=1e-12, max=1 - 1e-12),
                                  torch.zeros_like(cls_predicts_item).scatter(dim=-1,
                                                                              index=cls_target[:, None],
                                                                              value=1.0),
                                  self.alpha, self.gamma).sum(-1)
            reg_loss = torch.full_like(cls_loss, fill_value=INF)
            reg_loss[match_idx >= 0] = self.iou_loss(predicts_box, gt_boxes[bid][:, 1:][match_idx[match_idx >= 0]])
            combine_loss = reg_loss + cls_loss
            gt_idx = self.paa_assign(match_idx, combine_loss, len(gt_boxes[bid]), pred_num_per_layer)
            match_batch_idx.append(bid)
            match_anchor_idx.append((gt_idx >= 0).nonzero(as_tuple=False).squeeze(-1))
            match_gt_idx.append(gt_idx[gt_idx >= 0])
        all_cls_target = torch.zeros_like(cls_predicts)

        cls_batch_idx = sum([[i] * len(j) for i, j in zip(match_batch_idx, match_anchor_idx)], [])
        match_anchor_idx = torch.cat(match_anchor_idx)
        match_cls_idx = torch.cat([gt_boxes[i][:, 0][j] for i, j in zip(match_batch_idx, match_gt_idx)]).long()
        num_pos = len(match_cls_idx)
        all_cls_target[cls_batch_idx, match_anchor_idx, match_cls_idx] = 1.0
        all_cls_loss = focal_loss(cls_predicts.sigmoid(), all_cls_target, self.alpha, self.gamma).sum() / num_pos

        all_box_predicts = self.box_coder.decoder(reg_predicts[cls_batch_idx, match_anchor_idx],
                                                  all_anchors[match_anchor_idx])
        all_box_targets = torch.cat([gt_boxes[i][:, 1:][j] for i, j in zip(match_batch_idx, match_gt_idx)], dim=0)

        iou_targets = self.box_similarity(all_box_predicts.detach(), all_box_targets)
        all_iou_predicts = iou_predicts[cls_batch_idx, match_anchor_idx, 0]
        all_iou_loss = self.iou_loss_weight * self.bce(all_iou_predicts, iou_targets) / num_pos

        all_box_loss = self.reg_loss_weight * (self.iou_loss(
            all_box_predicts, all_box_targets) * iou_targets).sum() / iou_targets.sum()
        return all_cls_loss, all_box_loss, all_iou_loss, num_pos
