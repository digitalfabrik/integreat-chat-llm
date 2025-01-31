"""
Index pages for region & language
"""

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError
from integreat_chat.search.services.opensearch import OpenSearchSetup

class Command(BaseCommand):
    """
    Index pages for region & language
    """
    help = "Index pages for region & language"

    def add_arguments(self, parser):
        parser.add_argument("region", type=str)
        parser.add_argument("language", type=str)

    def handle(self, *args, **options):
        if "region" not in options or "language" not in options:
            raise CommandError('missing region or language argument')
        region_slug = options["region"]
        language_slug = options["language"]
        oss = OpenSearchSetup(password=settings.OPENSEARCH_PASSWORD)
        print(f"Indexing pages for region {options['region']} and language {options['language']}")
        print(oss.delete_index(f"{region_slug}_{language_slug}"))
        print(oss.create_index(f"{region_slug}_{language_slug}"))
        oss.index_pages(region_slug, language_slug)
