param(
    [string]$GradleVersion = "8.2",
    [string]$DistUrl = "https://services.gradle.org/distributions/gradle-8.2-bin.zip",
    [string]$TargetDir = "$PSScriptRoot\..\",
    [string]$WrapperDir = "$PSScriptRoot\..\gradle\wrapper\"
)

$ErrorActionPreference = 'Stop'

Write-Host "Regenerating Gradle wrapper using Gradle $GradleVersion..."

$zipPath = Join-Path $env:TEMP "gradle-$GradleVersion-bin.zip"

if (Test-Path $zipPath) { Remove-Item $zipPath -Force }

Write-Host "Downloading $DistUrl to $zipPath..."
Invoke-WebRequest -Uri $DistUrl -OutFile $zipPath -UseBasicParsing

Write-Host "Extracting gradle-wrapper.jar from archive..."
Add-Type -AssemblyName System.IO.Compression.FileSystem
$extractDir = Join-Path $env:TEMP "gradle_extract_$GradleVersion"
if (Test-Path $extractDir) { Remove-Item $extractDir -Recurse -Force }
[System.IO.Compression.ZipFile]::ExtractToDirectory($zipPath, $extractDir)

$libJar = Get-ChildItem -Path $extractDir -Recurse -Filter "gradle-wrapper.jar" | Select-Object -First 1
if (-not $libJar) {
    Write-Error "gradle-wrapper.jar not found in downloaded archive."
    exit 1
}

if (-not (Test-Path $WrapperDir)) { New-Item -ItemType Directory -Path $WrapperDir -Force | Out-Null }
Copy-Item -Path $libJar.FullName -Destination (Join-Path $WrapperDir "gradle-wrapper.jar") -Force

# Ensure gradle-wrapper.properties has correct distributionUrl
$gwProps = Join-Path $WrapperDir "gradle-wrapper.properties"
if (-not (Test-Path $gwProps)) {
    Write-Host "Creating gradle-wrapper.properties at $gwProps"
    "distributionBase=GRADLE_USER_HOME" | Out-File -FilePath $gwProps -Encoding UTF8
    "distributionPath=wrapper/dists" | Out-File -FilePath $gwProps -Encoding UTF8 -Append
    "zipStoreBase=GRADLE_USER_HOME" | Out-File -FilePath $gwProps -Encoding UTF8 -Append
    "zipStorePath=wrapper/dists" | Out-File -FilePath $gwProps -Encoding UTF8 -Append
    "distributionUrl=https\://services.gradle.org/distributions/gradle-$GradleVersion-bin.zip" | Out-File -FilePath $gwProps -Encoding UTF8 -Append
} else {
    (Get-Content $gwProps) -replace 'distributionUrl=.*', "distributionUrl=https://services.gradle.org/distributions/gradle-$GradleVersion-bin.zip" | Set-Content $gwProps
}

Write-Host "Gradle wrapper regenerated. Cleaning up temp files..."
Remove-Item $zipPath -Force
Remove-Item $extractDir -Recurse -Force

Write-Host "Done. You can now run .\gradlew.bat --version from the android folder."
