import ignite.metrics


class LossMetric(ignite.metrics.Loss):
    """Loss function metric with non fixed input tensor count"""
    @ignite.metrics.metric.reinit__is_reduced
    def update(self, output):
        average_loss = self._loss_fn(*output)
        if len(average_loss.shape) != 0:
            raise ValueError('loss_fn did not return the average loss.')

        N = self._batch_size(output[0])
        self._sum += average_loss.item() * N
        self._num_examples += N
