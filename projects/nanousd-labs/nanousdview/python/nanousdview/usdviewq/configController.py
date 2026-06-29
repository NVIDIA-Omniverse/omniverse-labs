#
# Copyright 2022 Pixar
#
# Licensed under the terms set forth in the LICENSE.txt file available at
# https://openusd.org/license.
#

from __future__ import print_function

import os
from functools import partial

from .common import SpawnNanousdview
from .qt import QtWidgets

class ConfigController:
    def __init__(self, currentConfig, appController):
        self._appController = appController
        self._currentConfig = currentConfig

        # nanousdview: always wire the File → State menu items. Pixar's stock
        # build hid them behind USDVIEWQ_CONFIG_CONTROLLER while still in
        # progress; in this fork they're known-working, so no env var gate.
        if os.getenv('USDVIEWQ_DISABLE_CONFIG_CONTROLLER'):
            self._hide(self._appController._ui)
        else:
            self.reloadConfigController()

    def _hide(self, ui):
        ui.menuLoad_New_State.menuAction().setVisible(False)
        ui.actionSave_State_To.setVisible(False)
        ui.menuSave_State_As.menuAction().setVisible(False)
        ui.actionSave_State_As_New_Config.setVisible(False)

    def reloadConfigController(self):
        """Refresh the File-menu state UI. Re-callable (clears the dynamic
        submenus before re-populating) so it can fire after a Save As New
        Config without producing duplicate entries."""
        ui = self._appController._ui
        if self._appController._configManager._configDirPath is None:
            self._hide(ui)
            return

        # Wire the static actions exactly once. Re-connecting on every reload
        # would stack callbacks (each save would fire N times after N reloads).
        if not getattr(self, "_actionsWired", False):
            ui.actionSave_State_To.triggered.connect(
                self._appController._configManager.save)
            ui.actionSave_State_As_New_Config.triggered.connect(
                lambda: self._saveAsTriggered())
            # XXX When no longer hiding controller behind env variable, move
            # this separator to the static ui file.
            ui.menuFile.insertSeparator(ui.actionQuit)
            self._actionsWired = True

        # Always offer one-click save. When launched without --config, the
        # ConfigManager treats "" as the default config — show that to the
        # user as "Save State To default".
        label = self._currentConfig if self._currentConfig else "default"
        save = ui.actionSave_State_To
        save.setText(f"Save State To {label}")
        save.setVisible(True)
        ui.actionSave_State_As_New_Config.setVisible(True)

        # Re-populate dynamic submenus from the current config list. clear()
        # drops both the actions and their triggered connections.
        configs = self._appController._configManager.getConfigs()[1:]

        ui.menuLoad_New_State.clear()
        if configs:
            for config in configs:
                ui.menuLoad_New_State.addAction(config).triggered.connect(
                    partial(
                        self._reopenWithConfig, config))
            ui.menuLoad_New_State.menuAction().setVisible(True)
        else:
            ui.menuLoad_New_State.menuAction().setVisible(False)

        ui.menuSave_State_As.clear()
        if configs:
            for config in configs:
                ui.menuSave_State_As.addAction(config).triggered.connect(
                    partial(self._appController._configManager.save, config))
            ui.menuSave_State_As.menuAction().setVisible(True)
        else:
            ui.menuSave_State_As.menuAction().setVisible(False)

    def _reopenWithConfig(self, config, checked=False):
        del checked
        SpawnNanousdview(
            self._appController._parserData.usdFile,
            ["--config", config])

    def _validateAndSaveConfig(self, newName, dialog):
        if not newName:
            print("Invalid config name, not saving", file=sys.stderr)
            return
        self._appController._configManager.save(newName)
        # Refresh the File-menu submenus so the new config appears in
        # "Reopen With State" and "Save State As" without a restart.
        self._currentConfig = newName
        self.reloadConfigController()
        dialog.close()

    def _saveAsTriggered(self):
        configDialog = QtWidgets.QDialog(self._appController._mainWindow)
        configDialog.setWindowTitle("Save State As")

        layout = QtWidgets.QHBoxLayout()
        field = QtWidgets.QLineEdit(self._currentConfig)
        field.textEdited.connect(lambda text: field.setText(text.lower()))
        layout.addWidget(field)

        fieldsLayout = QtWidgets.QVBoxLayout()
        fieldsLayout.addWidget(QtWidgets.QLabel(
            "Config Name"))
        fieldsLayout.addLayout(layout)
        fieldsLayout.addStretch()

        buttonBox = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Cancel | QtWidgets.QDialogButtonBox.StandardButton.Save)
        buttonBox.rejected.connect(configDialog.close)
        buttonBox.accepted.connect(lambda :
            self._validateAndSaveConfig(field.text(), configDialog))

        configDialogLayout = QtWidgets.QVBoxLayout()
        configDialogLayout.addLayout(fieldsLayout)
        configDialogLayout.addWidget(buttonBox)
        configDialog.setLayout(configDialogLayout)

        configDialog.open()
