# Copyright (c) OpenMMLab. All rights reserved.

from typing import Dict, List, Type, Union

from mmengine.dist import dist

from mmrazor.models.mutators.channel_mutator.channel_mutator import \
    ChannelMutator
from mmrazor.registry import MODELS
from mmrazor.utils import print_log
from .unit import GroupFisherChannelUnit


@MODELS.register_module()
class GroupFisherChannelMutator(ChannelMutator[GroupFisherChannelUnit]):
    """Channel mutator for GroupFisher Pruning Algorithm.

    Args:
        channel_unit_cfg (Union[dict, Type[ChannelUnitType]], optional):
            Config of MutableChannelUnits. Defaults to
            dict(type='GroupFisherChannelUnit',
                 default_args=dict(choice_mode='ratio')).
        parse_cfg (Dict): The config of the tracer to parse the model.
            Defaults to dict(type='ChannelAnalyzer',
                             demo_input=(1, 3, 224, 224),
                             tracer_type='FxTracer').
    """

    def __init__(self,
                 channel_unit_cfg: Union[dict,
                                         Type[GroupFisherChannelUnit]] = dict(
                                             type='GroupFisherChannelUnit'),
                 parse_cfg: Dict = dict(
                     type='ChannelAnalyzer',
                     demo_input=(1, 3, 224, 224),
                     tracer_type='FxTracer'),
                 **kwargs) -> None:
        super().__init__(channel_unit_cfg, parse_cfg, **kwargs)
        self.mutable_units: List[GroupFisherChannelUnit]

    def start_record_info(self) -> None:
        """Start recording the related information."""
        for unit in self.mutable_units:
            unit.start_record_fisher_info()

    def end_record_info(self) -> None:
        """Stop recording the related information."""
        for unit in self.mutable_units:
            unit.end_record_fisher_info()

    def reset_recorded_info(self) -> None:
        """Reset the related information."""
        for unit in self.mutable_units:
            unit.reset_recorded()

    def check_channels(self):
        total=0
        for unit in self.mutable_units:
            if unit.mutable_channel.activated_channels == 1:
                total+=1
        return total==len(self.mutable_units)

    def try_prune(self) -> None:
        """Prune the channel with the minimum fisher unless it is the last
        channel of the current layer."""
        min_imp = 1e5
        min_unit = self.mutable_units[0]
        min_channel_index = -1
        #  min_non_zero=None
        for unit in self.mutable_units:
            if unit.mutable_channel.activated_channels > 1:
                imp = unit.importance()
                nonzero = unit.mutable_channel.current_mask.nonzero()
                imp_nonzero=imp[nonzero]
                #  imp_mask=imp[mask]
                #  if imp_mask.isnan().any():
                if imp_nonzero.isnan().any():
                    if dist.get_rank() == 0:
                        print_log(
                            f'{unit.name} detects nan in importance, this pruning skips.'  # noqa
                        )
                    return
                #  if imp_mask.min() < min_imp:
                min, min_index = imp_nonzero.min(dim=0)
                if min < min_imp:
                    #  min_imp = imp.min().item()
                    min_imp = min
                    min_unit = unit
                    min_channel_index = nonzero[min_index].squeeze()
                    #  min_non_zero=imp[nonzero]

        if min_channel_index == -1:
            if self.check_channels():
                assert(0)
            else:
                if dist.get_rank() == 0:
                    print_log(f'min_imp={min_imp}')
                return

        assert min_channel_index >= 0
        #  assert min_unit.try_to_prune_min_channel(min_channel_index)
        if min_unit.try_to_prune_min_channel(min_channel_index):
            if dist.get_rank() == 0:
                #  print_log(
                    #  f'{min_unit.name} prunes a channel with min imp = {min_imp}, activated_channels={min_unit.mutable_channel.activated_channels}, number_of_channes={min_unit.num_channels}, channel_index = {min_channel_index}, min_non_zero={min_non_zero}'  # noqa
                #  )
                print_log(
                    f'{min_unit.name} prunes a channel with min imp = {min_imp}, activated_channels={min_unit.mutable_channel.activated_channels}, number_of_channes={min_unit.num_channels}, channel_index = {min_channel_index}'  # noqa
                )

    def update_imp(self) -> None:
        """Update the fisher information of each unit."""
        for unit in self.mutable_units:
            unit.update_fisher_info()

    def reset_imp(self) -> None:
        """Reset the fisher information of each unit."""
        for unit in self.mutable_units:
            unit.reset_fisher_info()
