# AgroVeyra repository backup (verifiable robocopy + a git bundle of the full history).
#
# Why this exists: the project has two trained, verified models and a long commit history living on a
# single disk with no git remote. This script copies the working tree to a second location, then
# proves the copy is complete (second robocopy pass must copy 0 files, and the model binaries are
# hash-compared). Datasets and re-installable environments are excluded by design - see $excludeDirs.
#
# Usage (from the repository root):
#   powershell -NoProfile -ExecutionPolicy Bypass -File ml\backup_repo.ps1
#   powershell -NoProfile -ExecutionPolicy Bypass -File ml\backup_repo.ps1 -Destination 'E:\backup'

param(
    [string]$Destination = 'C:\Users\smroh\OneDrive\AgriVeyra_backup'
)

$ErrorActionPreference = 'Continue'
$src = (Get-Location).Path
$worktree = Join-Path $Destination 'worktree'
$report = Join-Path $src 'ml\_backup_report.txt'
$bundle = Join-Path $Destination 'AgriVeyra_history.bundle'

New-Item -ItemType Directory -Force -Path $Destination | Out-Null
Remove-Item $report -ErrorAction SilentlyContinue

function Line($text) { $text | Out-File $report -Append -Encoding utf8; Write-Host $text }

Line ('=' * 100)
Line 'AgroVeyra backup'
Line ('=' * 100)
Line ("started      : " + (Get-Date).ToString('yyyy-MM-dd HH:mm:ss'))
Line ("source       : $src")
Line ("destination  : $Destination")
Line ""

# --- 1. what is being backed up, and what is deliberately excluded -------------------------------
Line 'free space on the destination drive:'
$destinationDrive = (Resolve-Path $Destination).Path.Substring(0, 3)
Line ('  ' + $destinationDrive + ' free = ' + [math]::Round((Get-PSDrive -Name $destinationDrive.Substring(0, 1)).Free / 1GB, 1) + ' GB')
Line ''
Line 'excluded from the file copy (regenerable, so they must not consume cloud quota):'
Line '  ml\data                 - datasets (IP102 is a download; the splits rebuild with the scripts)'
Line '  ml\.venv, ml\.export-libs, ml\.export-libs-314  - python environments (pip install)'
Line '  backend\.venv           - python environment (pip install)'
Line '  ml\runs, ml\export_build*  - training and export run output'
Line '  android\app\build, android\build, android\.gradle - gradle output (next build regenerates it)'
Line ''
Line 'included (the parts that cannot be regenerated): the source, the .git history, the deployed'
Line 'models, and every report/curve this session produced. robocopy prints the exact byte and file'
Line 'counts it copied in the summary below.'
Line ''

# --- 2. git bundle: the complete history in one verified file ------------------------------------
Line 'git bundle (full history, all refs):'
if (Test-Path $bundle) { Remove-Item $bundle -Force }
& git bundle create $bundle --all 2>&1 | Out-File $report -Append -Encoding utf8
if (Test-Path $bundle) {
    Line ('  wrote ' + $bundle + '  (' + [math]::Round((Get-Item $bundle).Length / 1MB, 2) + ' MB)')
    Line '  verify output:'
    & git bundle verify $bundle 2>&1 | ForEach-Object { Line ('    ' + $_) }
} else {
    Line '  BUNDLE FAILED - see the git output above'
}
Line ''

# --- 3. robocopy pass 1 --------------------------------------------------------------------------
$excludeDirs = @(
    # datasets and dead weight that must never consume cloud quota
    (Join-Path $src 'ml\data'),
    (Join-Path $src 'ml\dataset'),
    (Join-Path $src 'datasets'),
    (Join-Path $src '.tools'),
    (Join-Path $src '.kilo'),
    (Join-Path $src 'export-env'),
    (Join-Path $src '.venv_DEAD_DELETE_ME'),
    (Join-Path $src 'ml\runs'),
    (Join-Path $src 'ml\export_build'),
    (Join-Path $src 'ml\export_build_pest'),
    (Join-Path $src 'ml\weights'),
    (Join-Path $src 'ml\models'),
    (Join-Path $src 'backend\.venv'),
    (Join-Path $src 'android\app\build'),
    (Join-Path $src 'android\build'),
    (Join-Path $src 'android\.gradle'),
    (Join-Path $src 'ml\.venv'),
    (Join-Path $src 'ml\.export-libs'),
    (Join-Path $src 'ml\.export-libs-314'),
    # bare-name wildcards: these match by folder name anywhere in the tree.
    # 'dataset' and 'datasets' matter: ml\dataset holds ~8.8 GB of raw source images.
    'dataset',
    'datasets',
    'node_modules',
    '.venv',
    '.venv*',
    '.export-libs*',
    'export_build*',
    '__pycache__',
    '.pytest_cache',
    '.mypy_cache',
    '.gradle',
    'build'
)

$commonArgs = @('/E', '/COPY:DAT', '/DCOPY:DAT', '/R:2', '/W:2', '/MT:16', '/NP', '/NFL', '/NDL', '/XD') +
    $excludeDirs + @('/XF', '*.pyc')

Line 'clearing any previous (unpruned) copy at the destination:'
if (Test-Path $worktree) {
    Remove-Item $worktree -Recurse -Force -ErrorAction SilentlyContinue
    Line ('  worktree present after cleanup: ' + (Test-Path $worktree))
} else {
    Line '  nothing to clear'
}
Line ''

Line 'robocopy pass 1 (copy):'
& robocopy $src $worktree @commonArgs ("/LOG+:" + $report) | Out-Null
Line ('  robocopy exit code: ' + $LASTEXITCODE + '  (0-7 = success, 8+ = failure)')
Line ''

# --- 4. robocopy pass 2: must copy nothing ------------------------------------------------------
Line 'robocopy pass 2 (verify - the Files Copied count must be 0):'
& robocopy $src $worktree @commonArgs ("/LOG+:" + $report) | Out-Null
Line ('  robocopy exit code: ' + $LASTEXITCODE + '  (0 = nothing left to copy, which is what we want)')
Line ''

# --- 5. hash-compare the irreplaceable model binaries --------------------------------------------
Line 'model binaries (hash comparison source vs backup):'
$models = @(
    'android\app\src\main\assets\agroveyra_model.tflite',
    'android\app\src\main\assets\agroveyra_pest_model.tflite',
    'android\app\src\main\assets\agroveyra_triage_model.tflite',
    'backend\models\agroveyra_model.tflite',
    'backend\models\agroveyra_pest_model.tflite',
    'backend\models\agroveyra_triage_model.tflite'
)
foreach ($rel in $models) {
    $from = Join-Path $src $rel
    $to = Join-Path $worktree $rel
    if ((Test-Path $from) -and (Test-Path $to)) {
        $a = (Get-FileHash $from -Algorithm SHA256).Hash
        $b = (Get-FileHash $to -Algorithm SHA256).Hash
        Line ('  {0} {1}  {2}' -f $(if ($a -eq $b) { 'MATCH   ' } else { 'MISMATCH' }), $a.Substring(0, 16), $rel)
    } else {
        Line ('  MISSING  ' + $rel + '  (source exists: ' + (Test-Path $from) + ')')
    }
}
Line ''

# --- 6. reconciliation --------------------------------------------------------------------------
Line 'reconciliation:'
# Counting the source tree here would mean enumerating every dataset file (~300 k paths) only to
# filter them out again, which takes minutes. Completeness is already proven by pass 2 copying
# nothing, so this counts the copy instead and reports the git state of the backup.
$dstFiles = Get-ChildItem $worktree -Recurse -File -Force -ErrorAction SilentlyContinue
$dstSize = ($dstFiles | Measure-Object -Sum Length).Sum
Line ("  backup files   : {0}  ({1:N1} MB)" -f $dstFiles.Count, ($dstSize / 1MB))
if (Test-Path (Join-Path $worktree '.git')) {
    $head = & git -C $worktree log --oneline -1 2>&1
    Line ("  backup HEAD commit: $head")
    Line '  backup .git present: True'
} else {
    Line '  backup .git present: False (history not copied!)'
}
Line ''
Line ("finished     : " + (Get-Date).ToString('yyyy-MM-dd HH:mm:ss'))
Line ("report       : $report")
