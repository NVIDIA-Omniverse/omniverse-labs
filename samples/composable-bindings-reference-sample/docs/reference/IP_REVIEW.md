# Ongoing IP review

Maintainers must follow NVIDIA's [IP Review Process](https://nvidia.atlassian.net/wiki/spaces/OSS/pages/2529034695/IP+Review+Process)
before every public distribution of this project. This applies to ongoing
modifications and to all release material, including code, dependencies,
sample data, assets, screenshots and documentation. Review each commit if
pushing it distributes the change publicly; otherwise review the complete
release candidate before distribution.

## Review responsibility and criteria

The maintaining team selects reviewers with knowledge of the material.
Start with an author self-review. For a group review, assigned reviewers
must independently check the candidate. Ask the originating team to review
material that the maintaining team is not qualified to assess. OSRB does
not ordinarily perform the team's IP review.

The author and assigned reviewers must establish that:

1. Any patentable subject matter has been addressed with IP Legal and
   required patent filings have been made before release.
2. Copyright and license headers are correct, existing third-party
   attributions are preserved, and the distributed licenses and
   [third-party notices](../../THIRD_PARTY_NOTICES.md) match the actual
   release dependencies and included material.
3. The release does not reveal unannounced features or products.
4. It does not expose NVIDIA proprietary algorithms or internal information
   intended to remain confidential.
5. It does not give competitors a significant competitive advantage over
   NVIDIA products or services.
6. Any third-party code has the required OSRB guidance.

Keep a review record in the team's internal review system containing the
candidate commit or release, reviewed changes and artifacts, reviewer names,
date, findings against these criteria, and resolution of any open items.
Recheck subsequent changes before public distribution. This procedure also
applies if external contributions are accepted in the future; currently
the project does not accept them.

## Current verification status

The 2026-10-01 repository update supplies the policy's Apache-2.0 headers
for all 26 Python files and six launch scripts, the LICENSE copyright,
the contribution statement, and a dependency-based notice inventory.

These repository checks do not establish patent, confidentiality,
competitive-impact, or third-party approval decisions. A completed team
IP review record and confirmation of the ongoing reviewer responsibility
have not been supplied for this release candidate. The maintaining team
must record those decisions before public distribution. Adding this
procedure is not an assertion that IP review or OSRB release approval is
complete.
