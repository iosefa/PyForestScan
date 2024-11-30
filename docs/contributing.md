# Contributing to PyForestScan

Thank you for contributing to PyForestScan! Your involvement helps make this project a great tool for point cloud data processing and visualization of forest structure.

## Code of Conduct

By participating in this project, please follow our [Code of Conduct](CODE_OF_CONDUCT.md).

## How Can I Contribute?

### Reporting Bugs

- **Check Existing Issues** — Before opening a new bug report, see if the issue has already been reported. If it has, add any additional details in a comment.
  
- **Submit a Report** — If the issue hasn't been reported, open a new issue and fill out the provided template.

### Suggesting Enhancements

Have an idea to improve PyForestScan? Please open an issue to discuss your suggestion.

### Pull Requests

To contribute via pull requests:

1. **Fork the Repository** — Fork the PyForestScan repository and clone it locally.
2. **Create a Branch** — Make changes in a new branch. Use descriptive names like `feat/`, `fix/`, or `docs/` followed by the feature or fix name.
3. **Commit Your Changes** — Write a clear commit message describing your changes.
4. **Push to Your Fork** — Push the branch to your fork on GitHub.
5. **Create a Pull Request** — Open a pull request (PR) in the PyForestScan repository. Link any relevant issues.
6. **Code Review** — A maintainer will review your changes. You may need to make updates based on feedback.
7. **Merge** — Once approved, your PR will be merged into the main codebase.

## Style Guidelines

### Python

- Follow the [PEP 8](https://pep8.org/) style guide.
- Use type hints in functions.
- Add documentation to public APIs.

### Git Commit Messages

- Use present tense ("Add feature" not "Added feature").
- Limit the first line to 72 characters or fewer.
- Reference related issues and PRs when relevant.

## Releasing a New Version

### Steps for Creating a New Release

1. **Ensure `main` is up to date**:
   Confirm all changes intended for the release are merged into the `main` branch.

2. **Update the version**:
   Manually bump the version number in `setup.py` based on the type of release (major, minor, or patch) following [semantic versioning](https://semver.org/).

3. **Create a new tag**:
   Tag the release with the new version using the format `vX.X.X`. For example:
   ```bash
   git tag v1.2.0
   git push origin v1.2.0
   ```

4. **Deploy to PyPI**:
   The GitHub Actions workflow will automatically build and deploy the package to PyPI once the tag is pushed.

### Semantic Versioning Guidelines
- **Major version**: For incompatible API changes.
- **Minor version**: For backward-compatible features.
- **Patch version**: For backward-compatible bug fixes.

## Additional Notes

- Ensure compatibility with the latest dependencies.
- Update documentation when adding or changing features.
