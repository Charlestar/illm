[CmdletBinding()]
param(
    [string]$RepositoryRoot
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
    if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) {
        $RepositoryRoot = (Get-Location).Path
    }
    else {
        $RepositoryRoot = Split-Path -Parent $PSScriptRoot
    }
}

function Get-RepositoryRelativePath {
    param(
        [Parameter(Mandatory = $true)][string]$BasePath,
        [Parameter(Mandatory = $true)][string]$TargetPath
    )

    # Windows PowerShell 5.1 runs on .NET Framework, which does not expose
    # System.IO.Path.GetRelativePath.  All validated Markdown files live below
    # the repository root, so a normalized prefix removal is sufficient and
    # keeps this validator runnable on both Windows PowerShell and PowerShell 7.
    $normalizedBase = [System.IO.Path]::GetFullPath($BasePath).TrimEnd('\', '/')
    $normalizedTarget = [System.IO.Path]::GetFullPath($TargetPath)
    $prefix = $normalizedBase + [System.IO.Path]::DirectorySeparatorChar

    if ($normalizedTarget.StartsWith($prefix, [System.StringComparison]::OrdinalIgnoreCase)) {
        return $normalizedTarget.Substring($prefix.Length)
    }

    return $normalizedTarget
}

$specRoot = Join-Path $RepositoryRoot 'docs\specs'
$versionRoot = Join-Path $specRoot 'versions'
$issues = [System.Collections.Generic.List[string]]::new()

$releaseTrain = @(
    @{ File = 'v0.0-bootstrap-lab.md'; Target = 'v0.0.0'; Dependency = $null; Previous = 'none' },
    @{ File = 'v0.1-reference-runner.md'; Target = 'v0.1.0'; Dependency = 'v0.0.0'; Previous = './v0.0-bootstrap-lab.md' },
    @{ File = 'v0.2-stateful-streaming.md'; Target = 'v0.2.0'; Dependency = 'v0.1.0'; Previous = './v0.1-reference-runner.md' },
    @{ File = 'v0.3-batched-engine.md'; Target = 'v0.3.0'; Dependency = 'v0.2.0'; Previous = './v0.2-stateful-streaming.md' },
    @{ File = 'v0.4-paged-memory.md'; Target = 'v0.4.0'; Dependency = 'v0.3.0'; Previous = './v0.3-batched-engine.md' },
    @{ File = 'v0.5-preemptible-memory.md'; Target = 'v0.5.0'; Dependency = 'v0.4.0'; Previous = './v0.4-paged-memory.md' },
    @{ File = 'v0.6-chunked-prefill.md'; Target = 'v0.6.0'; Dependency = 'v0.5.0'; Previous = './v0.5-preemptible-memory.md' },
    @{ File = 'v0.7-prefix-aware-core.md'; Target = 'v0.7.0'; Dependency = 'v0.6.0'; Previous = './v0.6-chunked-prefill.md' },
    @{ File = 'v0.8-visual-replay.md'; Target = 'v0.8.0'; Dependency = 'v0.7.0'; Previous = './v0.7-prefix-aware-core.md' },
    @{ File = 'v1.0-stable-single-gpu.md'; Target = 'v1.0.0'; Dependency = 'v0.8.0'; Previous = './v0.8-visual-replay.md' },
    @{ File = 'v1.1-performance-season.md'; Target = 'v1.1.0'; Dependency = 'v1.0.0'; Previous = './v1.0-stable-single-gpu.md' },
    @{ File = 'v1.2-structured-outputs.md'; Target = 'v1.2.0'; Dependency = 'v1.1.0'; Previous = './v1.1-performance-season.md' },
    @{ File = 'v1.3-speculative-decoding.md'; Target = 'v1.3.0'; Dependency = 'v1.2.0'; Previous = './v1.2-structured-outputs.md' },
    @{ File = 'v2.0-single-node-tensor-parallel.md'; Target = 'v2.0.0'; Dependency = 'v1.1.0'; Previous = './v1.1-performance-season.md' },
    @{ File = 'v2.1-data-parallel-routing.md'; Target = 'v2.1.0'; Dependency = 'v2.0.0'; Previous = './v2.0-single-node-tensor-parallel.md' },
    @{ File = 'v3.0-disaggregated-runtime.md'; Target = 'v3.0.0'; Dependency = 'v2.1.0'; Previous = './v2.1-data-parallel-routing.md' },
    @{ File = 'v3.1-reliable-operations.md'; Target = 'v3.1.0'; Dependency = 'v3.0.0'; Previous = './v3.0-disaggregated-runtime.md' }
)

$requiredMetadata = @(
    'status',
    'spec_status',
    'release_stage',
    'qualification_tier',
    'performance_outcome',
    'lts_state',
    'spec_version',
    'target_release',
    'release_type',
    'depends_on',
    'project_plan',
    'previous_spec',
    'owners',
    'created',
    'last_updated'
)

$learningGoalHeading = '## 1. ' + (-join @(
    [char]0x672C,
    [char]0x7248,
    [char]0x672C,
    [char]0x5B66,
    [char]0x4E60,
    [char]0x76EE,
    [char]0x6807
))

$observableOutcomeHeading = '## 2. ' + (-join @(
    [char]0x7528,
    [char]0x6237,
    [char]0x53EF,
    [char]0x89C2,
    [char]0x5BDF,
    [char]0x7684,
    [char]0x6210,
    [char]0x679C
))

$requiredSections = @(
    $learningGoalHeading,
    $observableOutcomeHeading,
    '## 3. Scope',
    '## 4. Non-goals',
    '## 5. Architecture Delta',
    '## 6. Contracts',
    '## 7. Implementation Slices',
    '## 8. Correctness and Invariants',
    '## 9. Visualization Learning Story',
    '## 10. Performance Hypothesis',
    '## 11. Compatibility and Migration',
    '## 12. Definition of Done',
    '## 13. Known Limitations',
    '## Appendix A. Requirement Traceability',
    '## Appendix B. Open Questions'
)

$seenTargets = @{}
$seenRequirementIds = @{}

foreach ($release in $releaseTrain) {
    $path = Join-Path $versionRoot $release.File
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        $issues.Add("missing version spec: $($release.File)")
        continue
    }

    $content = Get-Content -LiteralPath $path -Raw -Encoding UTF8
    $frontmatterMatch = [regex]::Match($content, '(?s)\A#.*?\r?\n\r?\n```yaml\r?\n(?<body>.*?)\r?\n```')
    if (-not $frontmatterMatch.Success) {
        $issues.Add("$($release.File): missing first-block YAML metadata")
        $metadataContent = ''
    }
    else {
        $metadataContent = $frontmatterMatch.Groups['body'].Value
    }

    foreach ($field in $requiredMetadata) {
        if ($metadataContent -notmatch "(?m)^$([regex]::Escape($field)):\s*\S.*$") {
            $issues.Add("$($release.File): missing metadata '$field'")
        }
    }

    $metadataEnums = @{
        'spec_status' = @('draft', 'accepted', 'implemented', 'released', 'superseded')
        'release_stage' = @('dev_snapshot', 'release_candidate', 'released', 'withdrawn')
        'qualification_tier' = @('unqualified', 'learning_complete', 'functionally_qualified', 'stable_released')
        'performance_outcome' = @('not_applicable', 'pending', 'NoCandidate', 'Adopt', 'No-go')
        'lts_state' = @('not_applicable', 'candidate', 'promoted', 'eol')
    }
    foreach ($metadataField in $metadataEnums.Keys) {
        $metadataMatch = [regex]::Match($metadataContent, "(?m)^$([regex]::Escape($metadataField)):\s*(\S+)\s*$")
        if ($metadataMatch.Success -and $metadataMatch.Groups[1].Value -notin $metadataEnums[$metadataField]) {
            $issues.Add("$($release.File): invalid $metadataField '$($metadataMatch.Groups[1].Value)'")
        }
    }

    $statusMatch = [regex]::Match($metadataContent, '(?m)^status:\s*(\S+)')
    $specStatusMatch = [regex]::Match($metadataContent, '(?m)^spec_status:\s*(\S+)')
    if ($statusMatch.Success -and $specStatusMatch.Success -and $statusMatch.Groups[1].Value -ne $specStatusMatch.Groups[1].Value) {
        $issues.Add("$($release.File): status must mirror spec_status")
    }

    if ($metadataContent -notmatch '(?m)^qualification_profiles:\s*$') {
        $issues.Add("$($release.File): missing qualification_profiles list")
    }
    $profileResults = [regex]::Matches($metadataContent, '(?m)^\s{4}result:\s*(\S+)\s*$')
    if ($profileResults.Count -eq 0) {
        $issues.Add("$($release.File): qualification_profiles has no result")
    }
    foreach ($profileResult in $profileResults) {
        if ($profileResult.Groups[1].Value -notin @('not_run', 'pass', 'fail', 'invalid', 'not_run_resource_unavailable', 'not_applicable')) {
            $issues.Add("$($release.File): invalid qualification profile result '$($profileResult.Groups[1].Value)'")
        }
    }

    $targetMatch = [regex]::Match($metadataContent, '(?m)^target_release:\s*(\S+)\s*$')
    if (-not $targetMatch.Success) {
        $issues.Add("$($release.File): target_release is unreadable")
    }
    else {
        $target = $targetMatch.Groups[1].Value
        if ($target -ne $release.Target) {
            $issues.Add("$($release.File): target_release '$target' should be '$($release.Target)'")
        }
        if ($seenTargets.ContainsKey($target)) {
            $issues.Add("duplicate target_release '$target': $($seenTargets[$target]) and $($release.File)")
        }
        else {
            $seenTargets[$target] = $release.File
        }
    }

    $dependencyMatch = [regex]::Match($metadataContent, '(?m)^depends_on:\s*\[(?<items>[^\]]*)\]\s*$')
    if (-not $dependencyMatch.Success) {
        $issues.Add("$($release.File): depends_on must use an inline YAML list")
    }
    else {
        $dependencies = @($dependencyMatch.Groups['items'].Value.Split(',') | ForEach-Object { $_.Trim() } | Where-Object { $_.Length -gt 0 })
        if ($null -eq $release.Dependency) {
            if ($dependencies.Count -ne 0) {
                $issues.Add("$($release.File): depends_on must be empty")
            }
        }
        elseif ($dependencies.Count -ne 1 -or $dependencies[0] -ne $release.Dependency) {
            $issues.Add("$($release.File): depends_on must be exactly [$($release.Dependency)]")
        }
    }

    $previousMatch = [regex]::Match($metadataContent, '(?m)^previous_spec:\s*(\S+)\s*$')
    if (-not $previousMatch.Success -or $previousMatch.Groups[1].Value -ne $release.Previous) {
        $actualPrevious = if ($previousMatch.Success) { $previousMatch.Groups[1].Value } else { '(unreadable)' }
        $issues.Add("$($release.File): previous_spec '$actualPrevious' should be '$($release.Previous)'")
    }

    $configContract = [regex]::Match(
        $content,
        '(?ms)^### 6\.2 Config/manifest contract\s*$\r?\n(?<body>.*?)(?=^### 6\.3 )'
    )
    if ($configContract.Success) {
        $maturityRows = [regex]::Matches($configContract.Groups['body'].Value, '(?m)^\|[^|\r\n]+\|\s*(?<value>[^|\r\n]+?)\s*\|')
        foreach ($maturityRow in $maturityRows) {
            $maturity = $maturityRow.Groups['value'].Value.Trim().Trim('`')
            if ($maturity -eq '---' -or $maturity -match 'maturity|stability|' + [char]0x7A33 -or $maturity.Contains(([char]0x6210).ToString())) {
                continue
            }
            if ($maturity -notin @('experimental', 'format_frozen_for_v0', 'candidate', 'stable')) {
                $issues.Add("$($release.File): invalid contract maturity '$maturity'")
            }
        }
    }

    if ($release.Target -match '^v[123]') {
        $experimentCard = [regex]::Match(
            $content,
            '(?ms)^### 10\.2 Experiment card\s*$\r?\n(?<body>.*?)(?=^## 11\.)'
        )
        if ($experimentCard.Success) {
            $primaryMetric = [regex]::Match($experimentCard.Groups['body'].Value, '(?m)^primary_metric:\s*(\S+)\s*$')
            if (-not $primaryMetric.Success -or $primaryMetric.Groups[1].Value -notmatch '^illm\.[a-z0-9_.-]+\.v[0-9]+$') {
                $issues.Add("$($release.File): primary_metric must be a fully-qualified versioned illm metric ID")
            }
            $guardrailList = [regex]::Match($experimentCard.Groups['body'].Value, '(?m)^guardrails:\s*\[(?<items>[^\]]+)\]\s*$')
            if (-not $guardrailList.Success) {
                $issues.Add("$($release.File): guardrails must use an inline list of registry IDs")
            }
            else {
                foreach ($guardrail in $guardrailList.Groups['items'].Value.Split(',')) {
                    $metricId = $guardrail.Trim()
                    if ($metricId -notmatch '^illm\.[a-z0-9_.-]+\.v[0-9]+$') {
                        $issues.Add("$($release.File): guardrail '$metricId' is not a fully-qualified versioned illm metric ID")
                    }
                }
            }
        }
    }

    $lastHeadingIndex = -1
    foreach ($heading in $requiredSections) {
        $headingMatches = [regex]::Matches($content, "(?m)^$([regex]::Escape($heading))\s*$")
        if ($headingMatches.Count -eq 0) {
            $issues.Add("$($release.File): missing heading '$heading'")
        }
        elseif ($headingMatches.Count -gt 1) {
            $issues.Add("$($release.File): heading '$heading' appears more than once")
        }
        elseif ($headingMatches[0].Index -le $lastHeadingIndex) {
            $issues.Add("$($release.File): heading '$heading' is out of order")
        }
        else {
            $lastHeadingIndex = $headingMatches[0].Index
        }
    }

    $numberedSections = [regex]::Matches($content, '(?m)^## (\d+)\. .+$')
    if ($numberedSections.Count -ne 13) {
        $issues.Add("$($release.File): expected 13 numbered sections, found $($numberedSections.Count)")
    }
    foreach ($section in $numberedSections) {
        $bodyStart = $section.Index + $section.Length
        $nextHeading = [regex]::Match($content.Substring($bodyStart), '(?m)^## ')
        $bodyLength = if ($nextHeading.Success) { $nextHeading.Index } else { $content.Length - $bodyStart }
        $body = $content.Substring($bodyStart, $bodyLength).Trim()
        if ($body.Length -lt 20) {
            $issues.Add("$($release.File): section $($section.Groups[1].Value) appears empty")
        }
    }

    if ($content -match '<[A-Z][A-Z0-9_-]{2,}>|\bTODO\b|\bTBD\b|\bFIXME\b') {
        $issues.Add("$($release.File): contains unresolved placeholder/TODO")
    }

    $fenceCount = [regex]::Matches($content, '(?m)^```').Count
    if (($fenceCount % 2) -ne 0) {
        $issues.Add("$($release.File): has an unbalanced fenced code block")
    }

    if ($content -match '(?m)[ \t]+$') {
        $issues.Add("$($release.File): contains trailing whitespace")
    }

    if ($content.IndexOf([char]0xFFFD) -ge 0) {
        $issues.Add("$($release.File): contains a Unicode replacement character")
    }

    $traceability = [regex]::Match(
        $content,
        '(?ms)^## Appendix A\. Requirement Traceability\s*$\r?\n(?<body>.*?)(?=^## Appendix B\.)'
    )
    $requirementIdPattern = '(?m)^\|\s*(?<id>[A-Z][A-Z0-9]*(?:[-.][A-Z0-9]+)+)\s+[^|]+\|'
    $requirementMatches = if ($traceability.Success) {
        [regex]::Matches($traceability.Groups['body'].Value, $requirementIdPattern)
    }
    else {
        @()
    }
    if ($requirementMatches.Count -eq 0) {
        $issues.Add("$($release.File): traceability appendix has no recognizable requirement IDs")
    }
    foreach ($requirementMatch in $requirementMatches) {
        $requirementId = $requirementMatch.Groups['id'].Value
        if ($seenRequirementIds.ContainsKey($requirementId)) {
            $issues.Add("duplicate requirement ID '$requirementId': $($seenRequirementIds[$requirementId]) and $($release.File)")
        }
        else {
            $seenRequirementIds[$requirementId] = $release.File
        }
    }

    # Traceability must point to a concrete, in-range checkbox.  Human labels
    # such as "fault gate" drift when a checklist is edited, so the canonical
    # form is "§12.N item 3", "items 2, 4", or "items 3-7".
    $dodItemCounts = @{}
    $dodSection = [regex]::Match(
        $content,
        '(?ms)^## 12\. Definition of Done\s*$\r?\n(?<body>.*?)(?=^## 13\.)'
    )
    if ($dodSection.Success) {
        $dodSubsections = [regex]::Matches($dodSection.Groups['body'].Value, '(?m)^### 12\.(?<number>\d+) .+$')
        foreach ($dodSubsection in $dodSubsections) {
            $bodyStart = $dodSubsection.Index + $dodSubsection.Length
            $remaining = $dodSection.Groups['body'].Value.Substring($bodyStart)
            $nextSubsection = [regex]::Match($remaining, '(?m)^### 12\.\d+ .+$')
            $bodyLength = if ($nextSubsection.Success) { $nextSubsection.Index } else { $remaining.Length }
            $subsectionBody = $remaining.Substring(0, $bodyLength)
            $dodItemCounts[$dodSubsection.Groups['number'].Value] = [regex]::Matches($subsectionBody, '(?m)^- \[[ xX]\] ').Count
        }
    }

    if ($traceability.Success) {
        $traceRows = [regex]::Matches(
            $traceability.Groups['body'].Value,
            '(?m)^\|\s*(?<id>[A-Z][A-Z0-9]*(?:[-.][A-Z0-9]+)+)\s+[^|]+\|[^|]*\|[^|]*\|\s*(?<dod>[^|]+?)\s*\|'
        )
        foreach ($traceRow in $traceRows) {
            $requirementId = $traceRow.Groups['id'].Value
            $dodReference = $traceRow.Groups['dod'].Value.Trim()
            $sectionSignPattern = [regex]::Escape(([char]0x00A7).ToString())
            $dodReferencePattern = '^' + $sectionSignPattern + '12\.(?<section>\d+)\s+items?\s+(?<items>\d+(?:\s*-\s*\d+)?(?:\s*,\s*\d+(?:\s*-\s*\d+)?)*)$'
            $dodReferenceMatch = [regex]::Match($dodReference, $dodReferencePattern)
            if (-not $dodReferenceMatch.Success) {
                $issues.Add("$($release.File): requirement '$requirementId' must reference concrete DoD item numbers, found '$dodReference'")
                continue
            }

            $sectionNumber = $dodReferenceMatch.Groups['section'].Value
            if (-not $dodItemCounts.ContainsKey($sectionNumber) -or $dodItemCounts[$sectionNumber] -eq 0) {
                $issues.Add("$($release.File): requirement '$requirementId' references missing/empty §12.$sectionNumber")
                continue
            }

            $itemCount = [int]$dodItemCounts[$sectionNumber]
            foreach ($itemPartRaw in $dodReferenceMatch.Groups['items'].Value.Split(',')) {
                $itemPart = $itemPartRaw.Trim()
                $rangeMatch = [regex]::Match($itemPart, '^(?<start>\d+)\s*-\s*(?<end>\d+)$')
                if ($rangeMatch.Success) {
                    $start = [int]$rangeMatch.Groups['start'].Value
                    $end = [int]$rangeMatch.Groups['end'].Value
                    if ($start -gt $end) {
                        $issues.Add("$($release.File): requirement '$requirementId' has descending DoD range '$itemPart'")
                        continue
                    }
                    $itemsToCheck = $start..$end
                }
                else {
                    $itemsToCheck = @([int]$itemPart)
                }

                foreach ($itemNumber in $itemsToCheck) {
                    if ($itemNumber -lt 1 -or $itemNumber -gt $itemCount) {
                        $issues.Add("$($release.File): requirement '$requirementId' references §12.$sectionNumber item $itemNumber, but that subsection has $itemCount checkbox item(s)")
                    }
                }
            }
        }
    }

    $openQuestions = [regex]::Match(
        $content,
        '(?ms)^## Appendix B\. Open Questions\s*$\r?\n(?<body>.+)$'
    )
    if (-not $openQuestions.Success -or $openQuestions.Groups['body'].Value.Trim().Length -lt 20) {
        $issues.Add("$($release.File): open-questions appendix appears empty")
    }
}

$markdownFiles = Get-ChildItem -LiteralPath $specRoot -Filter '*.md' -File -Recurse
$forbiddenFileReferences = @(
    'v0.5-prefix-aware-core.md',
    'v0.6-visual-replay.md',
    'v2.0-single-node-multigpu.md'
)
foreach ($markdownFile in $markdownFiles) {
    $content = Get-Content -LiteralPath $markdownFile.FullName -Raw -Encoding UTF8
    foreach ($forbiddenReference in $forbiddenFileReferences) {
        if ($content.Contains($forbiddenReference)) {
            $relativeSource = Get-RepositoryRelativePath -BasePath $RepositoryRoot -TargetPath $markdownFile.FullName
            $issues.Add("$relativeSource`: contains obsolete spec reference '$forbiddenReference'")
        }
    }
    $linkMatches = [regex]::Matches($content, '\[[^\]]+\]\(([^)]+\.md)(?:#[^)]+)?\)')
    foreach ($linkMatch in $linkMatches) {
        $link = $linkMatch.Groups[1].Value
        if ($link -match '^(?:https?://|/)') {
            continue
        }
        $resolved = [System.IO.Path]::GetFullPath((Join-Path $markdownFile.DirectoryName $link))
        if (-not (Test-Path -LiteralPath $resolved -PathType Leaf)) {
            $relativeSource = Get-RepositoryRelativePath -BasePath $RepositoryRoot -TargetPath $markdownFile.FullName
            $issues.Add("$relativeSource`: broken local link '$link'")
        }
    }
}

$projectPlanPath = Join-Path $RepositoryRoot 'AI_INFERENCE_ENGINE_PROJECT_PLAN.md'
if (Test-Path -LiteralPath $projectPlanPath -PathType Leaf) {
    $projectPlanContent = Get-Content -LiteralPath $projectPlanPath -Raw -Encoding UTF8
    foreach ($forbiddenReference in $forbiddenFileReferences) {
        if ($projectPlanContent.Contains($forbiddenReference)) {
            $issues.Add("AI_INFERENCE_ENGINE_PROJECT_PLAN.md: contains obsolete spec reference '$forbiddenReference'")
        }
    }
    if ($projectPlanContent -match '(?im)^\|[^\r\n|]*structured output[^\r\n|]*V1\.1') {
        $issues.Add('AI_INFERENCE_ENGINE_PROJECT_PLAN.md: structured output workload must not be mapped to V1.1')
    }
    if ($projectPlanContent -match '(?im)^\|[^\r\n|]*speculative[^\r\n|]*V1\.1') {
        $issues.Add('AI_INFERENCE_ENGINE_PROJECT_PLAN.md: speculative workload must not be mapped to V1.1')
    }
    foreach ($masterPrimary in [regex]::Matches($projectPlanContent, '(?m)^\s*primary_metric:\s*(\S+)\s*$')) {
        if ($masterPrimary.Groups[1].Value -notmatch '^illm\.[a-z0-9_.-]+\.v[0-9]+$') {
            $issues.Add("AI_INFERENCE_ENGINE_PROJECT_PLAN.md: primary_metric '$($masterPrimary.Groups[1].Value)' is not a fully-qualified versioned illm metric ID")
        }
    }
    foreach ($masterInlineGuardrails in [regex]::Matches($projectPlanContent, '(?m)^\s*guardrails:\s*\[(?<items>[^\]]+)\]\s*$')) {
        foreach ($masterMetric in $masterInlineGuardrails.Groups['items'].Value.Split(',')) {
            $masterMetricId = $masterMetric.Trim()
            if ($masterMetricId -notmatch '^illm\.[a-z0-9_.-]+\.v[0-9]+$') {
                $issues.Add("AI_INFERENCE_ENGINE_PROJECT_PLAN.md: guardrail '$masterMetricId' is not a fully-qualified versioned illm metric ID")
            }
        }
    }
    foreach ($masterBlockGuardrails in [regex]::Matches($projectPlanContent, '(?ms)^\s*guardrails:\s*$\r?\n(?<items>(?:\s+-\s+\S+\s*\r?\n)+)')) {
        foreach ($masterGuardrailLine in [regex]::Matches($masterBlockGuardrails.Groups['items'].Value, '(?m)^\s+-\s+(\S+)\s*$')) {
            $masterMetricId = $masterGuardrailLine.Groups[1].Value
            if ($masterMetricId -notmatch '^illm\.[a-z0-9_.-]+\.v[0-9]+$') {
                $issues.Add("AI_INFERENCE_ENGINE_PROJECT_PLAN.md: guardrail '$masterMetricId' is not a fully-qualified versioned illm metric ID")
            }
        }
    }
}

$actualVersionFiles = @(Get-ChildItem -LiteralPath $versionRoot -Filter '*.md' -File)
if ($actualVersionFiles.Count -ne $releaseTrain.Count) {
    $issues.Add("expected $($releaseTrain.Count) version specs, found $($actualVersionFiles.Count)")
}

if ($issues.Count -gt 0) {
    Write-Host "SPEC validation failed with $($issues.Count) issue(s):" -ForegroundColor Red
    foreach ($issue in $issues) {
        Write-Host "- $issue" -ForegroundColor Red
    }
    exit 1
}

Write-Host "SPEC validation passed: $($releaseTrain.Count) version specs, complete sections, metadata, dependencies, traceability, contracts, and local links." -ForegroundColor Green
