"""
Index pages for region & language
"""

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError
from integreat_chat.search.services.opensearch import OpenSearchSetup
from integreat_chat.core.utils.integreat_cms import get_region_languages

class Command(BaseCommand):
    """
    Index pages for region & language
    """
    help = "Index pages for region & language"

    def add_arguments(self, parser):
        parser.add_argument("region", type=str)

    def handle(self, *args, **options):
        if "region" not in options:
            raise CommandError('missing region argument')
        region_slug = options["region"]
        oss = OpenSearchSetup(password=settings.OPENSEARCH_PASSWORD)
        for language_slug in get_region_languages(region_slug):
            self.stdout.write(self.style.SUCCESS(  # pylint: disable=no-member
                f"Indexing pages for region {region_slug} and language {language_slug}"
            ))
            oss.delete_index(f"{region_slug}_{language_slug}")
            oss.create_index(f"{region_slug}_{language_slug}")
            oss.index_pages(region_slug, language_slug)
