# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
<# Native Windows setup/recovery window. The SDK runs in its own child process. #>
param([string]$RuntimeRoot, [ValidateSet('public')][string]$Profile='public')
$ErrorActionPreference='Stop'
$repoRoot=Split-Path $PSScriptRoot -Parent
if (-not $RuntimeRoot) { $RuntimeRoot=Join-Path $repoRoot '.ov_library_lab_runtime' }
$RuntimeRoot=[IO.Path]::GetFullPath($RuntimeRoot)
New-Item -ItemType Directory -Force -Path (Join-Path $RuntimeRoot 'logs') | Out-Null
Add-Type -AssemblyName PresentationFramework
[xml]$markup=@'
<Window xmlns="http://schemas.microsoft.com/winfx/2006/xaml/presentation" Title="Omniverse Libraries Playground" Width="650" Height="440" WindowStartupLocation="CenterScreen" Background="#0F161D" Foreground="#E7EDEF">
 <StackPanel Margin="30">
  <TextBlock Text="Omniverse Libraries Playground" FontSize="23" Foreground="#A5E155" Margin="0,0,0,15"/>
  <TextBlock Text="The launcher prepares the public libraries, current campus and robot guide. First setup requires several GB of downloads." TextWrapping="Wrap" FontSize="15" Margin="0,0,0,18"/>
  <TextBlock Name="Status" Text="Ready to prepare the lab." TextWrapping="Wrap" FontSize="17" Height="80"/>
  <ProgressBar Name="Progress" Height="7" IsIndeterminate="False" Margin="0,0,0,18"/>
  <StackPanel Orientation="Horizontal">
   <Button Name="Start" Content="Start exploring" Padding="22,10" Margin="0,0,10,0"/>
   <Button Name="Repair" Content="Repair and retry" Padding="18,10" Margin="0,0,10,0"/>
   <Button Name="Logs" Content="View log" Padding="18,10"/>
  </StackPanel>
  <TextBlock Text="RTX GPU / Windows x64. Saved revisions stay in your runtime folder." TextWrapping="Wrap" FontSize="12" Margin="0,20,0,0" Foreground="#92A2AD"/>
 </StackPanel>
</Window>
'@
$window=[Windows.Markup.XamlReader]::Load((New-Object Xml.XmlNodeReader $markup))
$status=$window.FindName('Status'); $progress=$window.FindName('Progress')
$start=$window.FindName('Start'); $repair=$window.FindName('Repair'); $logs=$window.FindName('Logs')
$app=New-Object Windows.Application
$app.ShutdownMode='OnExplicitShutdown'
$script:worker=$null
$script:logPath=$null
$script:shown=$false
function Start-LabWorker([bool]$RepairInstall) {
    if ($script:worker -and -not $script:worker.HasExited) { return }
    $script:logPath=Join-Path $RuntimeRoot ('logs/launch-'+(Get-Date -Format 'yyyyMMdd-HHmmss')+'.log')
    $errorLog=$script:logPath+'.err'
    # Start-Process receives quoted argument values; quotes are rejected, never interpreted as shell code.
    foreach ($path in @($RuntimeRoot,$PSScriptRoot)) { if ($path.Contains('"')) { throw 'Paths containing quotation marks are unsupported.' } }
    $args=@('-NoProfile','-ExecutionPolicy','Bypass','-File',('"'+(Join-Path $PSScriptRoot 'customer_worker.ps1')+'"'),'-RuntimeRoot',('"'+$RuntimeRoot+'"'),'-Profile',$Profile)
    if ($RepairInstall) { $args+='-Repair' }
    $script:worker=Start-Process -FilePath powershell.exe -ArgumentList $args -WindowStyle Hidden -PassThru -RedirectStandardOutput $script:logPath -RedirectStandardError $errorLog
    # Retain the process handle: Windows PowerShell can otherwise lose ExitCode
    # when a redirected child exits before the next DispatcherTimer tick.
    $null=$script:worker.Handle
    $script:shown=$false
    $start.IsEnabled=$repair.IsEnabled=$false
    $progress.IsIndeterminate=$true
    $status.Text='Preparing your installation...'
}
$start.Add_Click({ Start-LabWorker $false })
$repair.Add_Click({ Start-LabWorker $true })
$logs.Add_Click({ if ($script:logPath) { Start-Process -FilePath notepad.exe -ArgumentList ('"'+$script:logPath+'"') } })
$timer=New-Object Windows.Threading.DispatcherTimer
$timer.Interval=[TimeSpan]::FromMilliseconds(250)
$timer.Add_Tick({
    if (-not $script:worker) { return }
    $lines=if (Test-Path -LiteralPath $script:logPath) { Get-Content -LiteralPath $script:logPath -Tail 80 } else { @() }
    $message=$lines | Where-Object { $_ -like 'LAB_PROGRESS:*' -or $_ -like 'LAB_ERROR:*' } | Select-Object -Last 1
    if ($message) { $status.Text=$message -replace '^LAB_(PROGRESS|ERROR):\s*','' }
    if (-not $script:shown -and ($lines -match '^LAB_READY:')) { $script:shown=$true; $window.Hide() }
    if ($script:worker.HasExited) {
        $script:worker.WaitForExit()
        $code=$script:worker.ExitCode
        $script:worker=$null
        $progress.IsIndeterminate=$false
        if ($code -eq 0) { $timer.Stop(); $app.Shutdown(); return }
        $window.Show(); $window.Activate()
        $start.Content='Retry launch'; $start.IsEnabled=$repair.IsEnabled=$true
        if (-not ($message -like 'LAB_ERROR:*')) { $status.Text="The lab exited unexpectedly (code $code). View the log, then retry or repair the installation." }
    }
})
$window.Add_Closing({
    if ($script:worker -and -not $script:worker.HasExited) {
        # This process handle belongs to this launch; include only its children.
        & taskkill.exe /PID $script:worker.Id /T /F | Out-Null
    }
    $timer.Stop(); $app.Shutdown()
})
$timer.Start()
$window.Add_ContentRendered({ Start-LabWorker $false })
[void]$app.Run($window)
