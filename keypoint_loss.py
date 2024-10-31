import torch
import torch.nn.functional as F

class KeypointLoss:
    def __init__(self, confidence_threshold=0.3):
        self.confidence_threshold = confidence_threshold

        # Define keypoint indices for different parts
        self.body_idxs = list(range(17))
        self.face_idxs = list(range(17, 85))
        self.left_hand_idxs = list(range(85, 106))
        self.right_hand_idxs = list(range(106, 127))
        self.foot_idxs = list(range(127, 133))

    def compute_loss(self, pred, target, keypoint_type='all'):
        """Compute the MSE loss between predicted and target keypoints."""
        losses = {}
        batch_size = pred['smplx_kpt'].shape[0]

        # Convert target keypoints to batch format if needed
        if len(target['body_keypoints'].shape) == 2:
            for k in target:
                if 'keypoints' in k:
                    target[k] = target[k].unsqueeze(0)

        if keypoint_type in ['all', 'body']:
            losses['body_2d'] = self._compute_keypoint_loss(
                pred['smplx_kpt'][:, self.body_idxs, :2],
                target['body_keypoints'][..., :2],
                target['body_keypoints'][..., 2]
            )

        if keypoint_type in ['all', 'face']:
            losses['face_2d'] = self._compute_keypoint_loss(
                pred['smplx_kpt'][:, self.face_idxs, :2],
                target['face_keypoints'][..., :2],
                target['face_keypoints'][..., 2]
            )

        if keypoint_type in ['all', 'hand']:
            losses['left_hand_2d'] = self._compute_keypoint_loss(
                pred['smplx_kpt'][:, self.left_hand_idxs, :2],
                target['lefthand_keypoints'][..., :2],
                target['lefthand_keypoints'][..., 2]
            )

            losses['right_hand_2d'] = self._compute_keypoint_loss(
                pred['smplx_kpt'][:, self.right_hand_idxs, :2],
                target['righthand_keypoints'][..., :2],
                target['righthand_keypoints'][..., 2]
            )

        total_loss = sum(losses.values())
        losses['total'] = total_loss

        return losses

    def _compute_keypoint_loss(self, pred, target, conf):
        """Compute the weighted MSE loss with refined scaling."""
        valid_mask = conf > self.confidence_threshold
        if valid_mask.sum() == 0:
            return torch.tensor(0., device=pred.device, requires_grad=True)

        # Use mean reduction instead of sum to prevent large values
        loss = F.mse_loss(pred[valid_mask], target[valid_mask], reduction='mean')

        # Weight the loss by confidence and take the mean
        weighted_loss = (loss * conf[valid_mask]).mean()

        return weighted_loss
