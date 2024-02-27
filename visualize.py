import matplotlib.pyplot as plt


def show_attn_weight(inputs, model, decoded_input, decoded_output):
    for layer in model.model.decoder.layers:
        weight = layer.attn_weight[0]
        # weight_sum = layer.attn_weight_sum[0].view(weight.shape[0], weight.shape[1], -1)
        decoded_input = decoded_input.rstrip()
        fig, axs = plt.subplots(2, 2)

        for i, attn_weight_head in enumerate(weight):
            attn_weight_head_nopad = attn_weight_head[:, :len(decoded_input)]

            ax = axs[(i % 4) // 2, (i % 4) % 2]
            ax.imshow(attn_weight_head_nopad.cpu().numpy(), aspect='auto', cmap='gray')
            ax.set_xticks(range(attn_weight_head_nopad.shape[1]))
            ax.set_yticks(range(len(decoded_output)))
            ax.set_xticklabels(decoded_input, fontsize=20)
            ax.set_yticklabels(decoded_output, fontsize=20)

            # ax = axs[i // 2, (i*2+1) % 4]
            # ax.imshow(weight_sum[i].cpu().numpy(), aspect='auto')
            # ax.set_xticks([])
            # ax.set_yticks([])

            if (i+1) % 4 == 0:
                # mappable = axs[0, 0].imshow(attn_weight_head.cpu().numpy())
                # plt.colorbar(mappable, ax=axs[0, 0])
                plt.show()
                fig, axs = plt.subplots(2, 2)

        # fig, axs = plt.subplots(2, 4)
        # fig.tight_layout()

        # for i, attn_weight_head in enumerate(weight):
        #     weight_sum = layer.attn_weight_sum[0].view(weight.shape[0], weight.shape[1], -1)

        #     ax = axs[i // 2, (i*2) % 4]
        #     ax.imshow(attn_weight_head.cpu().numpy(), aspect='auto')
        #     ax.set_xticks(range(len(inputs[0])))
        #     ax.set_yticks(range(len(inputs[0])))
        #     ax.set_xticklabels(decoded_input, fontsize=20)
        #     ax.set_yticklabels(decoded_input, fontsize=20)

        #     ax = axs[i // 2, (i*2+1) % 4]
        #     ax.imshow(weight_sum[i].cpu().numpy(), aspect='auto')
        #     # ax.set_yticks(range(weight_sum[i].shape[0]))
        #     # ax.set_yticklabels(decoded_input, fontsize=20)
        #     ax.set_yticks([])
        #     ax.set_xticks([])

        #     if (i+1) % 4 == 0:
        #         # mappable = axs[0, 0].imshow(attn_weight_head.cpu().numpy())
        #         # plt.colorbar(mappable, ax=axs[0, 0])
        #         fig.show()
        #         fig, axs = plt.subplots(2, 4)

            
        # fig, axs = plt.subplots(4, 4)
        # for i, attn_weight_head in enumerate(layer.attn_weight[0]):
        #     ax = axs[i // 4, i % 4]
        #     ax.imshow(attn_weight_head.cpu().numpy())
        #     ax.set_xticks(range(len(inputs[0])))
        #     ax.set_yticks(range(len(inputs[0])))
        #     ax.set_xticklabels(decode(inputs[0]).ljust(7))
        #     ax.set_yticklabels(decode(inputs[0]).ljust(7))

        #     if (i+1) % 16 == 0:
        #         print(i)
        #         mappable = axs[0, 0].imshow(attn_weight_head.cpu().numpy())  # Create a mappable object
        #         plt.colorbar(mappable, ax=axs[0, 0])  # Add colorbar
        #         plt.show()
        #         fig, axs = plt.subplots(4, 4)