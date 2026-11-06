# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A standalone OVUI panel: three widgets share one live equipment value."""
import asyncio
import omni.ui as ui

async def main():
    model = ui.SimpleFloatModel(.25)
    window = ui.Window('Operator controls', width=500, height=250)
    with window.frame:
        with ui.VStack(spacing=12):
            ui.Label('Fork height / shared value')
            ui.FloatSlider(model, min=0., max=1.)
            ui.FloatDrag(model, min=0., max=1., step=.01)
            ui.Button('Raise forks', clicked_fn=lambda: model.set_value(.85))
            status = ui.Label('Height 25%')
    def changed(value):
        status.text = f'Height {value.as_float:.0%}'
        # In the Lab, this same callback sets the fork transforms on ovstage.
        # Both the sidebar and equipment-anchored panel reuse this model.
    model.add_value_changed_fn(changed)
    await ui.run_async()

if __name__ == '__main__':
    ui.init('OVUI shared model', width=520, height=300)
    try:
        asyncio.run(main())
    finally:
        ui.shutdown()
