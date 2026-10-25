<#
.SYNOPSIS
    Start / manage the OpenCode bridge that SourceCheck AI uses for free models.

.DESCRIPTION
    The bridge exposes an OpenAI-compatible HTTP endpoint backed by your installed
    OpenCode CLI, so the agent can drive opencode zen's free models at zero cost.

        SourceCheck agent --HTTP--> 127.0.0.1:18383/v1 --> `opencode run` --> free models

.PARAMETER Action
    start     - run the bridge in this window (Ctrl+C to stop) [default]
    background- start it detached, so it survives this shell
    test      - health check + a live completion
    models    - list the free models the endpoint advertises
    stop      - stop a bridge listening on the configured port
    status    - is it up, and on which port?
    selfcheck - offline logic checks (no opencode, no network)

.EXAMPLE
    .\bridge\run-bridge.ps1
    .\bridge\run-bridge.ps1 test
    .\bridge\run-bridge.ps1 background
#>
[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [ValidateSet("start", "background", "test", "models", "stop", "status", "selfcheck")]
    [string]$Action = "start",

    [int]$Port = 18383,
    [string]$Model = "opencode/mimo-v2.6-flash-free"
)

$ErrorActionPreference = "Stop"
$bridgeScript = Join-Path $PSScriptRoot "opencode_bridge.py"
$baseUrl = "http://127.0.0.1:$Port"

function Get-BridgeProcess {
    $conn = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
    if (-not $conn) { return $null }
    $procId = $conn.OwningProcess | Select-Object -First 1
    return Get-Process -Id $procId -ErrorAction SilentlyContinue
}

function Test-Bridge {
    try {
        $health = Invoke-RestMethod -Uri "$baseUrl/health" -TimeoutSec 8
        Write-Host "[ok] bridge healthy" -ForegroundColor Green
        Write-Host "     version  : $($health.version)"
        Write-Host "     opencode : $($health.opencode_version)"
        Write-Host "     model    : $($health.model)"
        Write-Host "     in-flight: $($health.in_flight)"
    }
    catch {
        Write-Host "[fail] bridge not reachable at $baseUrl - run '.\bridge\run-bridge.ps1 start'" -ForegroundColor Red
        return $false
    }

    $body = @{ model = $Model; messages = @(@{ role = "user"; content = "Reply with exactly: bridge ok" }) } |
        ConvertTo-Json -Depth 5
    try {
        $resp = Invoke-RestMethod -Uri "$baseUrl/v1/chat/completions" -Method Post `
            -ContentType "application/json" -Body $body -TimeoutSec 180
        Write-Host "[ok] completion: $($resp.choices[0].message.content)" -ForegroundColor Green
        return $true
    }
    catch {
        $detail = $_.Exception.Message
        if ($_.Exception.Response) {
            $reader = New-Object System.IO.StreamReader($_.Exception.Response.GetResponseStream())
            $detail = $reader.ReadToEnd()
        }
        Write-Host "[fail] completion: $detail" -ForegroundColor Red
        return $false
    }
}

function Show-Models {
    try {
        $models = Invoke-RestMethod -Uri "$baseUrl/v1/models" -TimeoutSec 8
        Write-Host "Free models advertised by the bridge:" -ForegroundColor Cyan
        $models.data | ForEach-Object {
            $ctx = if ($_.context_length) { "  ctx=$($_.context_length)" } else { "" }
            Write-Host "  - $($_.id)$ctx"
        }
    }
    catch {
        Write-Host "[fail] could not fetch models: $($_.Exception.Message)" -ForegroundColor Red
    }
}

switch ($Action) {
    "selfcheck" {
        $env:BRIDGE_SELFCHECK = "1"
        python $bridgeScript
        Remove-Item Env:BRIDGE_SELFCHECK -ErrorAction SilentlyContinue
    }
    "status" {
        $proc = Get-BridgeProcess
        if ($proc) {
            Write-Host "[ok] bridge listening on $baseUrl (PID $($proc.Id))" -ForegroundColor Green
        }
        else {
            Write-Host "[--] nothing listening on port $Port" -ForegroundColor Yellow
        }
    }
    "stop" {
        $proc = Get-BridgeProcess
        if ($proc) {
            Stop-Process -Id $proc.Id -Force
            Write-Host "[ok] stopped bridge PID $($proc.Id)" -ForegroundColor Green
        }
        else {
            Write-Host "[--] nothing listening on port $Port" -ForegroundColor Yellow
        }
    }
    "test"     { if (Test-Bridge) { exit 0 } else { exit 1 } }
    "models"   { Show-Models }
    "background" {
        $proc = Start-Process -FilePath "python" -ArgumentList @($bridgeScript, "--port", $Port, "--model", $Model) `
            -WindowStyle Hidden -PassThru
        Start-Sleep -Seconds 4
        Write-Host "[ok] bridge started in background (PID $($proc.Id)) on $baseUrl" -ForegroundColor Green
        Write-Host "     stop it with: .\bridge\run-bridge.ps1 stop"
    }
    default {
        # Foreground: exit code and output are visible, Ctrl+C stops it.
        python $bridgeScript --port $Port --model $Model
    }
}
