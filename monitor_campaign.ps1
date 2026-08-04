# Campaign monitoring script - checks every 10 minutes
$logFile = "campaign_output.log"
$checkIntervalMinutes = 10

Write-Host "Campaign monitor started. Checking every $checkIntervalMinutes minutes..."
Write-Host "Monitoring log file: $logFile"
Write-Host ""

$iteration = 1
while ($true) {
    Write-Host "=== Monitor Check #$iteration at $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="
    
    # Check if campaign process is running
    $pythonProcs = Get-Process python -ErrorAction SilentlyContinue
    if ($pythonProcs) {
        Write-Host "Python processes running: $($pythonProcs.Count)"
        
        # Get last few lines of log
        if (Test-Path $logFile) {
            $lastLines = Get-Content $logFile -Tail 15 -ErrorAction SilentlyContinue
            Write-Host "`nLast 15 lines of campaign log:"
            Write-Host "----------------------------------------"
            $lastLines | ForEach-Object { Write-Host $_ }
            Write-Host "----------------------------------------"
        }
    } else {
        Write-Host "No Python processes found - campaign may have stopped!"
        
        # Show end of log
        if (Test-Path $logFile) {
            Write-Host "`nFinal lines of log:"
            Write-Host "----------------------------------------"
            Get-Content $logFile -Tail 30 | ForEach-Object { Write-Host $_ }
            Write-Host "----------------------------------------"
        }
        
        Write-Host "`nCampaign appears to have finished or crashed. Exiting monitor."
        break
    }
    
    Write-Host "`nSleeping for $checkIntervalMinutes minutes..."
    Write-Host ""
    
    Start-Sleep -Seconds ($checkIntervalMinutes * 60)
    $iteration++
}

Write-Host "`nMonitoring complete at $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
