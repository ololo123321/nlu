# старая функция для обучения
def train(
        self,
        examples_train: List[Example],
        examples_eval: List[Example],
        num_epochs: int = 1,
        batch_size: int = 128,
        train_op_name: str = "train_op",
        checkpoint_path: str = None,
        scope_to_save: str = None,
):
    train_loss = []

    # TODO: отделить конфигурацию оптимизатора от конфигурации обучения
    num_acc_steps = self.config["optimizer"]["num_accumulation_steps"]
    global_batch_size = batch_size * num_acc_steps
    epoch_steps = len(examples_train) // global_batch_size + 1
    num_train_steps = num_epochs * epoch_steps

    print(f"global batch size: {global_batch_size}")
    print(f"epoch steps: {epoch_steps}")
    print(f"num_train_steps: {num_train_steps}")

    train_op = getattr(self, train_op_name)

    epoch = 1
    best_score = -1
    num_steps_wo_improvement = 0
    num_lr_updates = 0

    if self.config["optimizer"]["reduce_lr_on_plateau"]:
        lr = self.config["optimizer"]["init_lr"]
    else:
        lr = None

    saver = None
    if checkpoint_path is not None:
        if scope_to_save is not None:
            var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope_to_save)
        else:
            var_list = tf.trainable_variables()
        saver = tf.train.Saver(var_list)

    for step in range(num_train_steps):
        examples_batch = random.sample(examples_train, batch_size)
        feed_dict, _ = self._get_feed_dict(examples_batch, mode=ModeKeys.TRAIN)
        _, loss = self.sess.run([train_op, self.loss], feed_dict=feed_dict)
        train_loss.append(loss)

        if step != 0 and step % epoch_steps == 0:

            print(f"epoch {epoch} finished. evaluation starts.")
            performance_info = self.evaluate(examples=examples_eval, batch_size=batch_size)
            score = performance_info["score"]

            if score > best_score:
                print("new best score:", score)
                best_score = score
                num_steps_wo_improvement = 0

                if saver is not None:
                    saver.save(self.sess, checkpoint_path)
                    print(f"saved new head to {checkpoint_path}")
            else:
                num_steps_wo_improvement += 1
                print("current score:", score)
                print("best score:", best_score)
                print("steps wo improvement:", num_steps_wo_improvement)

                if num_steps_wo_improvement == self.config["optimizer"]["max_steps_wo_improvement"]:
                    print("training finished due to max number of steps wo improvement encountered.")
                    break

                if self.config["optimizer"]["reduce_lr_on_plateau"]:
                    if num_steps_wo_improvement % self.config["optimizer"]["lr_reduce_patience"] == 0:
                        # TODO: restore best checkpoint here
                        lr_old = lr
                        lr *= self.config["optimizer"]["lr_reduction_factor"]
                        num_lr_updates += 1
                        print(f"lr reduced from {lr_old} to {lr}")

            if self.config["optimizer"]['custom_schedule']:
                lr = 1e-3
                if epoch < 100:
                    lr = 1e-3
                else:
                    lr = lr * 0.965 ** ((epoch - 100) + 1)

            lr = max(lr, self.config['optimizer']['min_lr'])

            print("lr:", lr)
            print("num lr updates:", num_lr_updates)
            print('=' * 50)

            epoch += 1