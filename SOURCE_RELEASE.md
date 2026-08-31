# Public source releases

This application is based on OASIS and is distributed under the GNU Affero
General Public License v3.0 only. Public releases in the companion source
repository contain the corresponding source for the version deployed as a
network service.

Each production release is identified by an immutable `source-<commit>` tag.
The release includes the backend and frontend source, database migrations,
tests, dependency lock files, container definitions, and deployment scripts
needed to build and run the application.

The source release intentionally does not contain credentials, environment
files with secrets, participant data, recordings, database contents, generated
evaluation transcripts, or internal research outputs. Those items are runtime
data or development artifacts rather than part of the deployed program's
corresponding source.

See [LICENSE](LICENSE) for the complete license text and [NOTICE](NOTICE) for
project attribution.
